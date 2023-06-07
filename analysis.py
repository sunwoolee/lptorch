from .major import get_distribution
from .modules import QBlock, QAdd, QClone
import matplotlib.pyplot as plt
import os

class distribution_info():
    def __init__(self, model):
        self.activation = {}
        self.parameter = {}
        
        self.get_activ_info(model)
        self.get_param_info(model)

    def get_activ_info(self, model):
        for n, m in model.named_modules():
            if isinstance(m, QBlock) or isinstance(m, QAdd) or isinstance(m, QClone):
                self.activation[n] = m.distribution

    def get_param_info(self, model):
        for n, p in model.named_parameters():
            dist_tmp = {}
            get_distribution(p, dist_tmp)
            self.parameter[n] = dist_tmp
    
    def visualize_distribution(self, file_path, target='both'):
        if target not in ['both', 'activation', 'parameter']:
            raise Exception('distribution target must be \'both\' or \'activation\' or \'parameter\'')

        if target in ['both','activation']:
            current = os.getcwd()
            path = os.path.join(file_path, 'activation')
            os.makedirs(path, exist_ok=True)
            os.chdir(path)
            for n, d in self.activation.items():
                for group, info in d.items():
                    if info:    # check empty dictionary
                        key = list(info.keys())
                        val = [e.item() for e in list(info.values())]
                        bar = plt.bar(key, val)
                        for rect in bar:
                            height = rect.get_height()
                            plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % height, ha='center', va='bottom')
                        plt.savefig(n+'_'+group+'.png')
                        plt.clf()
            os.chdir(current)
        if target in ['both','parameter']:
            current = os.getcwd()
            path = os.path.join(file_path, 'parameter')
            os.makedirs(path, exist_ok=True)
            os.chdir(path)
            for n, d in self.parameter.items():
                if d:   # check empty dictionary
                    key = list(d.keys())
                    val = [e.item() for e in list(d.values())]
                    bar = plt.bar(key, val)
                    for rect in bar:
                        height = rect.get_height()
                        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % height, ha='center', va='bottom')
                    plt.savefig(n+'.png')
                    plt.clf()
            os.chdir(current)
