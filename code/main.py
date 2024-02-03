import argparse
import os 
import imageio #images to gif
import pytorch3d
import numpy as np
from matplotlib import pyplot as plt

#starter code imports
from starter import render_mesh

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--q', type=float, default=1.1, help='Question number for the assignment')
    parser.add_argument("--output_path", type=str, default="data/shreyasj")

    #TODO: check question specific args
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--image_size", type=int, default=256)

    args = parser.parse_args()
    output_path = os.path.join(args.output_path, str(args.q))
    # Check if the output directory exists
    if not os.path.exists(output_path):
        # Create the directory if it doesn't exist
        os.makedirs(output_path)

    if args.q == 1.1:
        render360imageslist = []
        for i in range(-180, 180, 10):
            R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=3, elev=10, azim=i, degrees=True)
            image = render_mesh.render_cow(cow_path=args.cow_path, image_size=args.image_size, R=R, T=T)
            
            #convert the image to uint8 for compatibility with plt and imageio
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
            assert image.shape == (args.image_size, args.image_size, 3)

            #visualize the images
            plt.imshow(image)
            plt.title('Cow rendered at different angles')
            plt.show(block=False)
            plt.pause(0.05)

            render360imageslist.append(image)

        imageio.mimsave(os.path.join(output_path,'cow360.gif'), render360imageslist, fps=15)

    #if q is 1.2
    elif args.q == 1.2:
        pass
    #if q is 2.1
    elif args.q == 2.1:
        pass
    #if q is 2.2
    elif args.q == 2.2:
        pass
    #if q is 3
    elif args.q == 3:
        pass
    #if q is 4
    elif args.q == 4:
        pass
    elif args.q == 5.1:
        pass
    #if q is 5.2
    elif args.q == 5.2:
        pass
    #if q is 5.3
    elif args.q == 5.3:
        pass
    elif args.q == 6:
        pass
    elif args.q == 7:
        pass
    else:
        raise('Invalid question value')



