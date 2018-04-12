from freq_verbs import generate_verbs
from proc import generate_sents
from get_dataset import proc_wsd_data


dir_name = './data/masc/spoken/'
num_verbs = 6
generate_verbs(dir_name, num_verbs)

generate_sents()

proc_wsd_data()