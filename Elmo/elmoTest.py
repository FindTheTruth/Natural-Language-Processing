from allennlp.modules.elmo import Elmo,batch_to_ids
options_file ='elmo_2x4096_512_2048cnn_2xhighway_options.json'
weights_file = 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

num_output_representations = 2 # 2或者1

elmo = Elmo(options_file=options_file,weight_file=weights_file, num_output_representations=num_output_representations, dropout=0)

sentence_lists = [['I', 'have', 'a', 'dog', ',', 'it', 'is', 'so', 'cute','isn\'t it'],
                  ['I', 'love', 'Rong'],
                  ['an']]

character_ids = batch_to_ids(sentence_lists) #
print(character_ids, 'character_ids:', character_ids.shape) # [3,10,50]

res = elmo(character_ids)
print(len(res['elmo_representations']))  # 2
print(res['elmo_representations'][0].shape)  # [3, 10, 1024]
print(res['elmo_representations'][1].shape)  # [3, 10, 1024]

print(res['mask'])  # [3, 10] //mask的是有效的单词输入

