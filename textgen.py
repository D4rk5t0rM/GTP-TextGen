#https://www.youtube.com/watch?v=LjkubM5IIos
#https://github.com/minimaxir/gpt-2-simple
import gpt_2_simple as gpt2
import os
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
import datetime

def main():
    
    ##models:
    #model_name = "124M"
    #model_name = "355M"    
    #model_name = "774M"
    #model_name = "1558M"
    
    model_name = "355M"
    file_name = "champ.txt"

    
    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under ./models/124M/


    if not os.path.isfile(file_name):
        print("please provide a filename..")
        exit()        

    #GPU config
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction=0.77
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
    sess = tf.compat.v1.Session(config=config)
    
    #sess = gpt2.start_tf_sess() #old for CPU
    
    print('\n+++ Train model (y)? +++')
    train = input()
    if train == "" or train == "y" or train == 'yes':
        print('---> training model...\n')
        gpt2.finetune(sess,
                    file_name,
                    model_name=model_name,
                    steps=100)   # steps is max number of training steps - default: 1000
    else:
        print('---> not training model...\n')
    # gpt2.generate(sess) #generate session in file

    ## generate text to file
    gen_file = 'gpt2_gentext_{:%Y%m%d_%H%M%S}.txt'.format(datetime.datetime.now(datetime.timezone.utc))
    gpt2.generate_to_file(sess,
                      destination_path=gen_file,
                      length=10000,
                      temperature=0.7,
                      nsamples=1,
                      batch_size=1
                      )

main()