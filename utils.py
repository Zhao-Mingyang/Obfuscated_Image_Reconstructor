def get_features(image, model,discriminator, layers=None):
    """ Run an image forward through a model and get the features for 
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """
    
    ## TODO: Complete mapping layer names of PyTorch's VGGNet to names from the paper
    ## Need the layers for the content and style representations of an image
    if layers is None:
        layers = {'output':'Image_out',
            'layer4':'loss',
                  'fc': 'out'}
        
    features = {}
    x = image
    counter=0
    total=0
    # model._modules is a dictionary holding each module in the model


    x = ResnetGenerator.cuda(x)
    features[layers['output']] = x

     
    for name, layer in discriminator._modules.items():
        if name=='fc':
            x=x.view(1, -1)
#             print(x.size())
        x = layer(x)
#         print(x.size())
        
        if name in layers:
            features[layers[name]] = x
            
    return features


def get_targetfeatures(image, discriminator, layers=None):
    """ Run an image forward through a model and get the features for 
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """
    
    ## TODO: Complete mapping layer names of PyTorch's VGGNet to names from the paper
    ## Need the layers for the content and style representations of an image
    if layers is None:
        layers = {'output':'Image_out',
            'layer4':'loss',
                  'fc': 'out'}
        
    features = {}
    x = image
   
     
    for name, layer in discriminator._modules.items():
        if name=='fc':
            x=x.view(1, -1)
#             print(x.size())
        x = layer(x)
#         print(x.size())
        
        if name in layers:
            features[layers[name]] = x
            
    return features