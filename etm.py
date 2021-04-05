import torch
import torch.nn.functional as F 
import numpy as np 
import math 
from data import get_batch
from pathlib import Path
from gensim.models.fasttext import FastText as FT_gensim
from utils import nearest_neighbors, get_topic_coherence, get_topic_diversity

from torch import nn, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ETM(nn.Module):
    def __init__(self, num_topics, vocab_size, t_hidden_size, rho_size, emsize, 
                    theta_act, embeddings=None, train_embeddings=True, enc_drop=0.5):
        super(ETM, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.emsize = emsize
        self.t_drop = nn.Dropout(enc_drop)

        self.theta_act = self.get_activation(theta_act)
        
        ## define the word embedding matrix \rho
        if train_embeddings:
            self.rho = nn.Linear(rho_size, vocab_size, bias=False)
        else:
            num_embeddings, emsize = embeddings.size()
            rho = nn.Embedding(num_embeddings, emsize)
            self.rho = embeddings.clone().float().to(device)

        ## define the matrix containing the topic embeddings
        self.alphas = nn.Linear(rho_size, num_topics, bias=False)
    
        ## define variational distribution for \theta_{1:D} via amortizartion
        print(vocab_size, " THE Vocabulary size is here ")
        self.q_theta = nn.Sequential(
                nn.Linear(vocab_size, t_hidden_size), 
                self.theta_act,
                nn.Linear(t_hidden_size, t_hidden_size),
                self.theta_act,
            )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act 

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def encode(self, bows):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta

    def get_beta(self):
        """
        This generate the description as a defintion over words

        Returns:
            [type]: [description]
        """
        try:
            logit = self.alphas(self.rho.weight) # torch.mm(self.rho, self.alphas)
        except:
            logit = self.alphas(self.rho)
        beta = F.softmax(logit, dim=0).transpose(1, 0) ## softmax over vocab dimension
        return beta

    def get_theta(self, normalized_bows):
        """
        getting the topic poportion for the document passed in the normalixe bow or tf-idf"""
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1) 
        return theta, kld_theta

    def decode(self, theta, beta):
        """compute the probability of topic given the document which is equal to theta^T ** B

        Args:
            theta ([type]): [description]
            beta ([type]): [description]

        Returns:
            [type]: [description]
        """
        res = torch.mm(theta, beta)
        almost_zeros = torch.full_like(res, 1e-6)
        results_without_zeros = res.add(almost_zeros)
        predictions = torch.log(results_without_zeros)
        return predictions

    def forward(self, bows, normalized_bows, theta=None, aggregate=True):
        ## get \theta
        if theta is None:
            theta, kld_theta = self.get_theta(normalized_bows)
        else:
            kld_theta = None

        ## get \beta
        beta = self.get_beta()

        ## get prediction loss
        preds = self.decode(theta, beta)
        recon_loss = -(preds * bows).sum(1)
        if aggregate:
            recon_loss = recon_loss.mean()
        return recon_loss, kld_theta

    def get_optimizer(self, args):
        """
        Get the model default optimizer 

        Args:
            sefl ([type]): [description]
        """
        if args.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.wdecay)
        elif args.optimizer == 'adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=args.lr, weight_decay=args.wdecay)
        elif args.optimizer == 'adadelta':
            optimizer = optim.Adadelta(self.parameters(), lr=args.lr, weight_decay=args.wdecay)
        elif args.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(self.parameters(), lr=args.lr, weight_decay=args.wdecay)
        elif args.optimizer == 'asgd':
            optimizer = optim.ASGD(self.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
        else:
            print('Defaulting to vanilla SGD')
            optimizer = optim.SGD(self.parameters(), lr=args.lr)
        self.optimizer = optimizer
        return optimizer


    def train_for_epoch(self, epoch, args, training_set):
        """
        train the model for the given epoch 

        Args:
            epoch ([type]): [description]
        """
        self.train()
        acc_loss = 0
        acc_kl_theta_loss = 0
        cnt = 0
        number_of_docs = torch.randperm(args.num_docs_train)
        batch_indices = torch.split(number_of_docs, args.batch_size)
        print("The number of the indices I am using for the training is ", len(batch_indices))
        for idx, indices in enumerate(batch_indices):
            print("Running for ", idx)
            self.optimizer.zero_grad()
            self.zero_grad()
            data_batch = get_batch(training_set, indices, device)
            normalized_data_batch = data_batch
            recon_loss, kld_theta = self.forward(data_batch, normalized_data_batch)
            total_loss = recon_loss + kld_theta
            total_loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), args.clip)
            self.optimizer.step()

            acc_loss += torch.sum(recon_loss).item()
            acc_kl_theta_loss += torch.sum(kld_theta).item()
            cnt += 1
            if idx % args.log_interval == 0 and idx > 0:
                cur_loss = round(acc_loss / cnt, 2) 
                cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
                cur_real_loss = round(cur_loss + cur_kl_theta, 2)

                print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                    epoch, idx, len(indices), self.optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))

        cur_loss = round(acc_loss / cnt, 2) 
        cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
        cur_real_loss = round(cur_loss + cur_kl_theta, 2)
        print('*'*100)
        print('Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                epoch, self.optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
        print('*'*100)

    
    def visualize(self, args, vocabulary, show_emb=False):
        Path.cwd().joinpath("results").mkdir(parents=True, exist_ok=True)
        self.eval()
        model_path = Path.home().joinpath("Projects", 
                                        "Personal", 
                                        "balobi_nini", 
                                        'models', 
                                        'embeddings_one_gram_fast_tweets_only').__str__()
        model_gensim = FT_gensim.load(model_path)

        # need to update this .. 
        queries = ['felix', 'covid', 'pprd', '100jours', 'beni', 'adf', 'muyembe', 'fally']

        ## visualize topics using monte carlo
        results_file_name = "topic_results_{}_{}.txt".format(args.batch_size, args.epochs)
        results_file_name = Path.cwd().joinpath("results", results_file_name)
        with torch.no_grad():
            print('#'*100)
            print('Visualize topics...')
            topics_words = []
            gammas = self.get_beta()
            for k in range(args.num_topics):
                gamma = gammas[k]
                top_words = list(gamma.cpu().numpy().argsort()[-args.num_words+1:][::-1])
                topic_words = [vocabulary[a].strip() for a in top_words]
                topics_words.append(' '.join(topic_words))
                with open(results_file_name, "a") as results_file:
	                results_file.write('Topic {}: {}\n'.format(k, topic_words))
            with open(results_file_name, "a") as results_file:
	            results_file.write(10*'#'+'\n') # But this could have been done as a function

            if show_emb:
                ## visualize word embeddings by using V to get nearest neighbors
                print('#'*100)
                print('Visualize word embeddings by using output embedding matrix')
                try:
                    embeddings = self.rho.weight  # Vocab_size x E
                except:
                    embeddings = self.rho         # Vocab_size x E
                neighbors = []
                for word in queries:
                    print('word: {} .. neighbors: {}'.format(
                        word, nearest_neighbors(model_gensim, word)))
                print('#'*100)


    def evaluate(self, args, source, training_set, vocabulary , test_1, test_2, tc=False, td=False):
        """
        Compute perplexity on document completion.
        """
        self.eval()
        with torch.no_grad():
            if source == 'val':
                indices = torch.split(torch.tensor(range(args.num_docs_valid)), args.eval_batch_size)
            else: 
                indices = torch.split(torch.tensor(range(args.num_docs_test)), args.eval_batch_size)

            ## get \beta here
            beta = self.get_beta()

            ### do dc and tc here
            acc_loss = 0
            cnt = 0
            indices_1 = torch.split(torch.tensor(range(args.num_docs_test_1)), args.eval_batch_size)
            for idx, indice in enumerate(indices_1):
                data_batch_1 = get_batch(test_1, indice, device)
                sums_1 = data_batch_1.sum(1).unsqueeze(1)
                if args.bow_norm:
                    normalized_data_batch_1 = data_batch_1 / sums_1
                else:
                    normalized_data_batch_1 = data_batch_1
                theta, _ = self.get_theta(normalized_data_batch_1)
                ## get predition loss using second half
                data_batch_2 = get_batch(test_2, indice, device)
                sums_2 = data_batch_2.sum(1).unsqueeze(1)
                res = torch.mm(theta, beta)
                preds = torch.log(res)
                recon_loss = -(preds * data_batch_2).sum(1)
                loss = recon_loss / sums_2.squeeze()
                loss = np.nanmean(loss.numpy())
                acc_loss += loss
                cnt += 1
            cur_loss = acc_loss / cnt
            ppl_dc = round(math.exp(cur_loss), 1)
            print('*'*100)
            print('{} Doc Completion PPL: {}'.format(source.upper(), ppl_dc))
            print('*'*100)
            if tc or td:
                beta = beta.data.cpu().numpy()
                if tc:
                    print('Computing topic coherence...')
                    get_topic_coherence(beta, training_set, vocabulary)
                if td:
                    print('Computing topic diversity...')
                    get_topic_diversity(beta, 25)
            return ppl_dc
