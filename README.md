# Flower-Recognition-System
Deep learning desktop application that accurately predicts the species of a flower in a given image. It utilizes Transfer Learning with Google's MobileNetV2  for the image processing and feature recognition, and a Python GUI library for the interface. The user inputs an image of a flower from the 102 flower categories listed below and the application will output the top 5 predictions for the flower.

# Available Flowers for recognition 

The model is trained on the Oxford 102 Flower Dataset and can recognize the following species:

Alpine Sea Holly, Anthurium, Artichoke, Azalea, Balloon Flower, Ball Moss, Barbeton Daisy, Bearded Iris, Bee Balm, Bishop Of Llandaff, Black-Eyed Susan, Blackberry Lily, Blanket Flower, Bolero Deep Blue, Bougainvillea, Bromelia, Buttercup, Californian Poppy, Camellia, Canna Lily, Canterbury Bells, Carnation, Cape Flower, Cautleya Spicata, Cherry Blossom, Clematis, Colt'S Foot, Columbine, Common Dandelion, Corn Poppy, Cyclamen, Daffodil, Desert-Rose, English Marigold, Fire Lily, Foxglove, Frangipani, Fritillary, Garden Phlox, Gaura, Gazania, Geranium, Giant White Arum Lily, Globe-Flower, Globe Thistle, Grape Hyacinth, Great Masterwort, Hard-Leaved Pocket Orchid, Hibiscus, Hippeastrum, Iris, Japanese Anemone, King Protea, Lenten Rose, Lily, Lotus, Love In The Mist, Magnolia, Mallow, Marigold, Mexican Aster, Mexican Petunia, Monkshood, Moon Orchid, Morning Glory, Orchid, Osteospermum, Oxeye Daisy, Orange Dahlia, Passion Flower, Pelargonium, Peruvian Lily, Petunia, Pincushion Flower, Pink Primrose, Poinsettia, Primula, Prince Of Wales Feathers, Purple Coneflower, Red Ginger, Rose, Ruby-Lipped Cattleya, Siam Tulip, Silverbush, Snapdragon, Spear Thistle, Spring Crocus, Stemless Gentian, Sunflower, Sweet Pea, Sweet William, Sword Lily, Thorn Apple, Tiger Lily, Toad Lily, Tree Mallow, Tree Poppy, Trumpet Creeper, Tulip, Wallflower, Water Lily, Watercress, Wild Pansy, Windflower, Yellow Iris

# Model Training

The project uses Transfer Learning for the model training. It uses Google's highly accurate pre-trained model as a base and customizes it to the specific flower dataset. The training goes as follows:

1. Before reaching the model, each image is resized to the same standard of 224x224 and augmented (either flipped, zoomed or sheared) as to make sure the model learns the actual structure of the flowers, not the images given specifically.

