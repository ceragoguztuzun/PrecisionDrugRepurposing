from model.tools import *
from ordered_set import OrderedSet
from torch.utils.data import DataLoader
from model.data_loader import *
from model.predict import *
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from sklearn.metrics import roc_curve, auc


class Main(object):

	def __init__(self, params):
		self.p = params
		self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

		self.logger.info(vars(self.p))
		pprint(vars(self.p))

		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')

		self.load_data()
		self.model        = self.add_model()
		self.optimizer    = torch.optim.Adam(self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)

	def load_data(self):
		ent_set, rel_set = OrderedSet(), OrderedSet()
		tt_file = self.p.test_file.split('.')[0]
		for split in ['train', 'valid']:
			for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
				sub, rel, obj = map(str.lower, line.strip().split('\t'))
				ent_set.add(sub)
				rel_set.add(rel)
				ent_set.add(obj)

		self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
		self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
		
		self.id2node = {idx: ent.upper() for idx, ent in enumerate(ent_set)}
		self.id2link = {idx: rel.upper() for idx, rel in enumerate(rel_set)}
		
		self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})

		self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
		self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

		self.p.num_ent		= len(self.ent2id)
		self.p.num_rel		= len(self.rel2id) // 2
		self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim


		self.data	= ddict(list)
		sr2o		= ddict(set)

		for split in ['train', 'test', 'valid', tt_file]:
			for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
				sub, rel, obj = map(str.lower, line.strip().split('\t'))
				sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
				self.data[split].append((sub, rel, obj))

				if split == 'train': 
					sr2o[(sub, rel)].add(obj)
					sr2o[(obj, rel+self.p.num_rel)].add(sub)
		self.data = dict(self.data)

		self.sr2o = {k: list(v) for k, v in sr2o.items()}
		for split in ['test', 'valid',tt_file]:
			for sub, rel, obj in self.data[split]:
				sr2o[(sub, rel)].add(obj)
				sr2o[(obj, rel+self.p.num_rel)].add(sub)

		self.sr2o_all = {k: list(v) for k, v in sr2o.items()}

		self.triples = ddict(list)

		for (sub, rel), obj in self.sr2o.items():
			self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				rel_inv = rel + self.p.num_rel
				self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 	   'label': self.sr2o_all[(sub, rel)]})
				self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

		for split in [tt_file]:
			for sub, rel, obj in self.data[split]:
				rel_inv = rel + self.p.num_rel
				self.triples['{}'.format(split)].append({'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
				
		self.triples = dict(self.triples)

		def get_data_loader(dataset_class, split, batch_size, shuffle=False):
			return  DataLoader(
					dataset_class(self.triples[split], self.p),
					batch_size      = batch_size,
					shuffle         = shuffle,
					num_workers     = max(0, self.p.num_workers),
					collate_fn      = dataset_class.collate_fn
				)

		self.data_iter = {
			'train'		:   get_data_loader(TrainDataset, 'train', 	self.p.batch_size),
			'valid_head'	:   get_data_loader(TestDataset,  'valid_head', self.p.batch_size),
			'valid_tail'	:   get_data_loader(TestDataset,  'valid_tail', self.p.batch_size),
			'test_head'	:   get_data_loader(TestDataset,  'test_head',  self.p.batch_size),
			'test_tail'	:   get_data_loader(TestDataset,  'test_tail',  self.p.batch_size),
			tt_file	:   get_data_loader(TestDataset,  tt_file,  self.p.batch_size),
		}

		self.chequer_perm	= self.get_chequer_perm()
		self.edge_index, self.edge_type = self.construct_adj()

	def construct_adj(self):
		edge_index, edge_type = [], []

		for sub, rel, obj in self.data['train']:
			edge_index.append((sub, obj))
			edge_type.append(rel)

		# Adding inverse edges
		for sub, rel, obj in self.data['train']:
			edge_index.append((obj, sub))
			edge_type.append(rel + self.p.num_rel)

		edge_index	= torch.LongTensor(edge_index).to(self.device).t()
		edge_type	= torch.LongTensor(edge_type). to(self.device)

		return edge_index, edge_type

	def get_chequer_perm(self):
		ent_perm  = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.perm)])
		rel_perm  = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.perm)])

		comb_idx = []
		for k in range(self.p.perm):
			temp = []
			ent_idx, rel_idx = 0, 0

			for i in range(self.p.k_h):
				for j in range(self.p.k_w):
					if k % 2 == 0:
						if i % 2 == 0:
							temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
							temp.append(rel_perm[k, rel_idx]+self.p.embed_dim); rel_idx += 1;
						else:
							temp.append(rel_perm[k, rel_idx]+self.p.embed_dim); rel_idx += 1;
							temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
					else:
						if i % 2 == 0:
							temp.append(rel_perm[k, rel_idx]+self.p.embed_dim); rel_idx += 1;
							temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
						else:
							temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
							temp.append(rel_perm[k, rel_idx]+self.p.embed_dim); rel_idx += 1;

			comb_idx.append(temp)

		chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
		return chequer_perm

	def add_model(self):
		model = GPKG_PREDICT(self.edge_index, self.edge_type, self.chequer_perm, params=self.p)
		model.to(self.device)
		return model

	def read_batch(self, batch, split):
		if split == 'train':
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label, None, None
		else:
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label

	def load_model(self, load_path):
		state				= torch.load(load_path)
		state_dict			= state['state_dict']
		self.best_val_mrr 		= state['best_val']['mrr']
		self.best_val 			= state['best_val']

		self.model.load_state_dict(state_dict)
		self.optimizer.load_state_dict(state['optimizer'])

	def calculate_precision_recall_at_k(self, predictions, labels, K_values):
		precision_at_k = {k: [] for k in K_values}
		recall_at_k = {k: [] for k in K_values}

		for pred, label in zip(predictions, labels):

			# Ensure pred and label are 2D
			pred = pred.unsqueeze(0) if pred.dim() == 1 else pred
			label = label.unsqueeze(0) if label.dim() == 1 else label

			for k in K_values:
				topk_preds = torch.topk(pred, k, dim=1)[1]
				true_positives = torch.sum(torch.gather(label, 1, topk_preds), dim=1).float()
				precision_at_k[k].extend((true_positives / k).tolist())
				total_positives = torch.sum(label, dim=1).float()
				recall_at_k[k].extend((true_positives / total_positives.clamp(min=1)).tolist())

		avg_precision_at_k = {k: np.mean(precision_at_k[k]) for k in K_values}
		avg_recall_at_k = {k: np.mean(recall_at_k[k]) for k in K_values}

		return avg_precision_at_k, avg_recall_at_k



	def testpredict(self, split=''):
		save_path = os.path.join('./model_saved', self.p.name.replace(':', '-'))

		self.load_model(save_path)
		self.logger.info('Successfully Loaded previous model')
			
		self.model.eval()
		split = self.p.test_file.split('.')[0]

		all_preds = []
		all_labels = []
		K_values = [1, 3, 5, 10]  # Different K values
		precision_at_k = {k: [] for k in K_values}
		recall_at_k = {k: [] for k in K_values}

		with torch.no_grad():
			results = {}
			train_iter = iter(self.data_iter['{}'.format(split)])

			for step, batch in enumerate(train_iter):
				sub, rel, obj, label	= self.read_batch(batch, split)
				pred			= self.model.forward(sub, rel, None)
				b_range			= torch.arange(pred.size()[0], device=self.device)
				target_pred		= pred[b_range, obj]
				#pred 			= torch.where(label.byte(), torch.zeros_like(pred), pred)
				pred[b_range, obj] 	= target_pred
				ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]

				ranks 			= ranks.float()

				# Collecting predictions and labels
				all_preds.append(pred.cpu().numpy())
				all_labels.append(label.cpu().numpy())

				prelist = torch.argsort(pred, dim=1, descending=False)
				prelist = pred

				f=open(self.p.save_result,'a')
				for x in b_range.cpu().numpy() :
				    prelistpro = float(pred[x][obj[x]].cpu().numpy())
				    curlist = pred[x].cpu().numpy()
				    num = np.sum(curlist > prelistpro)  
				    f.write(str(self.id2ent[int(sub[x].cpu().numpy())])+'\t'+str(self.id2rel[int(rel[x].cpu().numpy())])+'\t'+str(self.id2ent[int(obj[x].cpu().numpy())])+'\t'+str(num+1)+'\t'+str(prelistpro)+'\n')
				f.close()				
				
				results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
				results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
				results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
				for k in range(10):
					results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

				if step % 100 == 0:
					all_preds_flat = np.concatenate(all_preds).ravel()
					all_labels_flat = np.concatenate(all_labels).ravel()

					self.logger.info('[{}, Step {}]\t{}'.format(split.title(),  step, self.p.name))
					print('->> [{}, Step {}]\t{}'.format(split.title(),  step, self.p.name))
					self.logger.info(f'Results: {results}')

			all_preds_flat = [torch.from_numpy(pred) for pred in np.concatenate(all_preds)]
			all_labels_flat = [torch.from_numpy(label) for label in np.concatenate(all_labels)]
			K_values = [1, 3, 5, 10, 15, 20]

			# Calculate precision and recall at k
			avg_precision, avg_recall = self.calculate_precision_recall_at_k(all_preds_flat, all_labels_flat, K_values)
			results['avg_prec@k'] = avg_precision
			results['avg_recall@k'] = avg_recall

		print(results)
		return results

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Parser For Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	# Dataset and Experiment name
	parser.add_argument('-data',           dest='dataset',         default='FB15k-237',            		help='Dataset to use for the experiment')
	parser.add_argument('-name',            			default='testrun_'+str(uuid.uuid4())[:8],	help='Name of the experiment')

	# Training parameters
	parser.add_argument('-gpu',		type=str,               default='0',					help='GPU to use, set -1 for CPU')
	parser.add_argument('-neg_num',        dest='neg_num',         default=1000,    	type=int,       	help='Number of negative samples to use for loss calculation')
	parser.add_argument('-batch',          dest='batch_size',      default=128,    	type=int,       	help='Batch size')
	parser.add_argument('-l2',		type=float,             default=0.0,					help='L2 regularization')
	parser.add_argument('-lr',		type=float,             default=0.001,					help='Learning Rate')
	parser.add_argument('-num_workers',	type=int,               default=10,                      		help='Maximum number of workers used in DataLoader')
	parser.add_argument('-seed',           dest='seed',            default=41504,   		type=int,       	help='Seed to reproduce results')
	parser.add_argument('-restore',   	dest='restore',       	action='store_true',            		help='Restore from the previously saved model')
	parser.add_argument('-opn',             dest='opn',             default='corr',                 help='Composition Operation to be used in CompGCN')
	
	# Model parameters
	parser.add_argument('-lbl_smooth',     dest='lbl_smooth',	default=0.1,		type=float,		help='Label smoothing for true labels')
	parser.add_argument('-embed_dim',	type=int,              	default=None,                   		help='Embedding dimension for entity and relation, ignored if k_h and k_w are set')
	parser.add_argument('-num_bases',	dest='num_bases', 	default=-1,   	type=int, 	help='Number of basis relation vectors to use')
	parser.add_argument('-init_dim',	dest='init_dim',	default=100,	type=int,	help='Initial dimension size for entities and relations')
	parser.add_argument('-gcn_dim',	  	dest='gcn_dim', 	default=200,   	type=int, 	help='Number of hidden units in GCN')
	parser.add_argument('-gcn_layer',	dest='gcn_layer', 	default=1,   	type=int, 	help='Number of GCN Layers to use')
	parser.add_argument('-gcn_drop',	dest='dropout', 	default=0.1,  	type=float,	help='Dropout to use in GCN Layer')

	parser.add_argument('-bias',      	dest='bias',          	action='store_true',            		help='Whether to use bias in the model')
	parser.add_argument('-form',		type=str,               default='plain',            			help='The reshaping form to use')
	parser.add_argument('-k_w',	  	dest='k_w', 		default=10,   		type=int, 		help='Width of the reshaped matrix')
	parser.add_argument('-k_h',	  	dest='k_h', 		default=20,   		type=int, 		help='Height of the reshaped matrix')
	parser.add_argument('-num_filt',  	dest='num_filt',      	default=200,     	type=int,       	help='Number of filters in convolution')
	parser.add_argument('-ker_sz',    	dest='ker_sz',        	default=7,     		type=int,       	help='Kernel size to use')
	parser.add_argument('-perm',      	dest='perm',          	default=1,      	type=int,       	help='Number of Feature rearrangement to use')
	parser.add_argument('-hid_drop',  	dest='hid_drop',      	default=0.3,    	type=float,     	help='Dropout for Hidden layer')
	parser.add_argument('-hid_drop2',  	dest='hid_drop2', 	default=0.3,  	type=float,	help='ConvE: Hidden dropout')
	parser.add_argument('-feat_drop', 	dest='feat_drop',     	default=0.3,    	type=float,     	help='Dropout for Feature')
	parser.add_argument('-inp_drop',  	dest='inp_drop',      	default=0.1,    	type=float,     	help='Dropout for Input layer')
	parser.add_argument('-save_result',  	default='pre_result',    	    	help='save file directory')
	parser.add_argument('-test_file',      default='',    	     	help='test file name')

	parser.add_argument('-scalars',  	default=None,    	type=str,     	help='Filename of scalars tensor to be used for modulator in forward embedding pass')

	# Logging parameters
	parser.add_argument('-logdir',    	dest='log_dir',       	default='./log/',               		help='Log directory')
	parser.add_argument('-config',    	dest='config_dir',    	default='./config/',            		help='Config directory')
	

	args = parser.parse_args()
	
	set_gpu(args.gpu)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	model = Main(args)
	model.testpredict()
