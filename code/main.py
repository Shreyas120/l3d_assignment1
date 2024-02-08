import argparse
import os 
import imageio #images to gif
import torch
import pytorch3d
import numpy as np
from matplotlib import pyplot as plt
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.io import load_obj
import mcubes

#starter code imports
from starter import render_mesh
from starter import dolly_zoom
from starter import utils
from starter import camera_transforms
from starter import render_generic

def render360GIFfromPCL(pcl,name,output_path,image_size=512, viz=True):
    render360imageslist = []
    renderer = utils.get_points_renderer(image_size=image_size, device=device, background_color=(1, 1, 1))

    for i in range(-180, 180, 10):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=20, elev=0, azim=i, degrees=True)
        R_fix = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float() #Flip upside down

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R@R_fix, T=T, fov=60, device=device)

        rend = renderer(pcl, cameras=cameras)
        image = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)

        #convert the image to uint8 for compatibility with plt and imageio
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        if viz:
        #visualize the images
            plt.imshow(image)
            plt.title(name + ' rendered at different angles')
            plt.show(block=False)
            plt.pause(0.05)

        render360imageslist.append(image)

    print('Saving the 360 GIF for ' + name + ' in ' + output_path)
    imageio.mimsave(os.path.join(output_path,name+'.gif'), render360imageslist, fps=15)

def render360GIFfromMesh(mesh,name,output_path,image_size=512,viz=True):
    render360imageslist = []
    # Place a point light in front of the object
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
    renderer = utils.get_mesh_renderer(image_size=image_size)

    for i in range(-180, 180, 10):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=10, elev=10, azim=i, degrees=True)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

        rend = renderer(mesh, cameras=cameras, lights=lights)
        image = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)

        #convert the image to uint8 for compatibility with plt and imageio
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        if viz:
        #visualize the images
            plt.imshow(image)
            plt.title(name + ' rendered at different angles')
            plt.show(block=False)
            plt.pause(0.05)

        render360imageslist.append(image)

    print('Saving the 360 GIF for ' + name + ' in ' + output_path)
    imageio.mimsave(os.path.join(output_path,name+'.gif'), render360imageslist, fps=15)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--q', type=float, default=1.1, help='Question number for the assignment')
    parser.add_argument("--output_path", type=str, default="data/shreyasj")

    #TODO: check question specific args
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--duration", type=float, default=3)

    args = parser.parse_args()
    output_path = os.path.join(args.output_path, str(args.q))
    # Check if the output directory exists
    if not os.path.exists(output_path):
        # Create the directory if it doesn't exist
        os.makedirs(output_path)
    
    device = utils.get_device()

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

    elif args.q == 1.2:
        dolly_zoom.dolly_zoom(
        image_size=args.image_size,
        num_frames=args.num_frames,
        duration=args.duration,
        output_file=os.path.join(output_path,'dolly_zoom.gif')
        )

    elif args.q == 2.1:
        vertices = torch.tensor([[2.5, 2.5, 5], [0, 0, 0], [2.5, 5, 0], [5, 0, 0]], dtype=torch.float32)
        vertices = vertices.unsqueeze(0)
        faces = torch.tensor([[1, 2, 3], [1, 3, 0], [1, 0, 2], [0, 2, 3]], dtype=torch.int64)
        faces = faces.unsqueeze(0)

        textures = torch.ones_like(vertices)  # (1, N_v, 3)
        textures = textures * torch.tensor([0.7, 0.7, 1])  # (1, N_v, 3)
        mesh = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures),
        )
        mesh = mesh.to(device)
        mesh = pytorch3d.structures.Meshes(verts=vertices,faces=faces,textures=pytorch3d.renderer.TexturesVertex(textures))
        mesh = mesh.to(device)
        render360GIFfromMesh(mesh,'tetrahedron',output_path,args.image_size,viz=True)

    elif args.q == 2.2:
        vertices = torch.tensor([[ 1, -1, -1],   #0
                                 [-1, -1, -1],   #1
                                 [-1,  1, -1],   #2
                                 [ 1,  1, -1],   #3
                                 [ 1, -1,  1],   #4
                                 [-1, -1,  1],   #5
                                 [ 1,  1,  1],   #6
                                 [-1,  1,  1]],  #7
                                 dtype=torch.float32)
        vertices = vertices.unsqueeze(0)
        faces = torch.tensor([[0, 3, 2], [2, 1, 0], 
                              [4, 6, 3], [3, 0, 4],
                              [5, 7, 2], [2, 1, 5], 
                              [4, 6, 7], [7, 5, 4],
                              [6, 7, 2], [7, 2, 3], 
                              [0, 4, 5], [5, 1, 0]], 
                              dtype=torch.int64)
        faces = faces.unsqueeze(0)

        textures = torch.ones_like(vertices)  # (1, N_v, 3)
        textures = textures * torch.tensor([0.2, 0.2, 1])  # (1, N_v, 3)
        mesh = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures),
        )
        mesh = mesh.to(device)
        mesh = pytorch3d.structures.Meshes(verts=vertices,faces=faces,textures=pytorch3d.renderer.TexturesVertex(textures))
        mesh = mesh.to(device)
        render360GIFfromMesh(mesh,'cube',output_path,args.image_size,viz=True)

    elif args.q == 3:
        # Get the renderer.
        renderer = utils.get_mesh_renderer(image_size=args.image_size)

        # Get the vertices, faces, and textures.
        vertices, faces = utils.load_cow_mesh()
        vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
        faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)\

        textures = torch.ones_like(vertices)  # (1, N_v, 3)
        ## Extract the z-coordinates
        z_coords = vertices[:, :, 2]
        # Find the max and min z-coordinate
        z_max = z_coords.max()
        z_min = z_coords.min()

        color1, color2 = torch.tensor([1,1,0]), torch.tensor([0,1,1])
        alpha = (z_coords - z_min) / (z_max - z_min)
        alpha = alpha.unsqueeze(-1)
        textures = alpha * color2 + (1 - alpha) * color1

        mesh = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures),
        )
        mesh = mesh.to(device)
        render360GIFfromMesh(mesh,'Texture',output_path,args.image_size,viz=True)

    elif args.q == 4:

        # Load cow with axis mesh
        mesh = pytorch3d.io.load_objs_as_meshes(["data/cow_with_axis.obj"]).to(device)

        #Place lights in front of the cow
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

        cam_tfs = [(None,None)]*4 #(R_relative (XYZ Euler Angles), T_relative) 
        
        #0 Rotate clockwise -> Rotate camera frame anti-clockwise to match world frame 
        cam_tfs[0] = ([0, 0, np.pi/2],[0, 0, 0])

        #1 World origin further away from camera 
        cam_tfs[1] = ([0,0, 0],[0, 0, 3])
        
        #2 World origin moves left in the camera frame
        cam_tfs[2] = ([0, 0, 0],[0.5, 0, 0])
        
        #3 Side view
        cam_tfs[3] = ([0, -np.pi/2, 0],[3, 0, 3])
        
        R, T = [], []
        for cam_tf in cam_tfs:
            R_relative, T_relative = cam_tf
            R_relative = pytorch3d.transforms.euler_angles_to_matrix(torch.tensor(R_relative), "XYZ").float()
            T_relative = torch.tensor(T_relative).float()

            R_effective = (R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]))
            T.append(R_relative @ torch.tensor([0.0, 0, 3]) + T_relative)
            R.append(R_effective.T)
           
        # Stack all camera transforms along new dimension
        R, T = torch.stack(R, dim=0), torch.stack(T, dim=0)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R,
                                                           T=T,
                                                           device=device)

        # renderer = utils.get_mesh_renderer(image_size=args.image_size)
        # img = renderer(mesh, cameras=cameras, lights=lights).cpu().numpy()[0, ..., :3]
        # img = np.clip(img * 255, 0, 255).astype(np.uint8)
        # plt.imshow(img)
        # plt.show()

        #Visualize the camera transforms and object
        # plot_scene({"All Views": {"Mesh": mesh, "Cameras": cameras}}).show()
        
        # Render and save the images
        renderer = utils.get_mesh_renderer(image_size=args.image_size)
        images = renderer(mesh.extend(4), cameras=cameras, lights=lights)  
        for idx,img in enumerate(images):
            img = img.cpu().numpy()
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            plt.imshow(img)
            plt.show()
            plt.imsave(os.path.join(output_path, (str(idx)+'.png')),img)

    elif args.q == 5.1:
        data = render_generic.load_rgbd_data()

        points_1, rgb_1 = utils.unproject_depth_image(image=torch.tensor(data['rgb1']), mask=torch.tensor(data['mask1']), depth=torch.tensor(data['depth1']), camera=data['cameras1'])
        points_1, rgb_1 = points_1.unsqueeze(0), rgb_1.unsqueeze(0)
        pcl_1 = pytorch3d.structures.Pointclouds(points=points_1, features=rgb_1).to(device)

        points_2, rgb_2 = utils.unproject_depth_image(image=torch.tensor(data['rgb2']), mask=torch.tensor(data['mask2']), depth=torch.tensor(data['depth2']), camera=data['cameras2'])
        points_2, rgb_2 = points_2.unsqueeze(0), rgb_2.unsqueeze(0)
        pcl_2 = pytorch3d.structures.Pointclouds(points=points_2, features=rgb_2).to(device)

        pcl = pytorch3d.structures.join_pointclouds_as_scene([pcl_1, pcl_2])

        #render the point clouds
        render360GIFfromPCL(pcl_1,'pcl1',output_path,args.image_size,viz=False)
        render360GIFfromPCL(pcl_2,'pcl2',output_path,args.image_size,viz=False)
        render360GIFfromPCL(pcl,'pcl',output_path,args.image_size,viz=False)

    elif args.q == 5.2:
        #Parametric equation for Torus (https://web.maths.unsw.edu.au/~rsw/Torus/index.php#:~:text=Torus%20parametrization&text=z%20%3D%20R2%20sin(v,%5B0%2C%202%20Pi).&text=(%20R1%20%2D%20(x2,is%20R1%20%2F%20R2.)
        R1, R2 = 1, 0.5 
        u = torch.linspace(0, 2 * np.pi, args.num_samples)
        v = torch.linspace(0, 2 * np.pi, args.num_samples)
        
        # Densely sample phi and theta on a grid
        U, V = torch.meshgrid(u, v)

        x = (R1 + R2*torch.cos(V) ) * torch.cos(U)
        y = (R1 + R2*torch.cos(V) ) * torch.sin(U)
        z =  R2 * torch.sin(V)
        points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
        color = (points - points.min()) / (points.max() - points.min())

        taurus_pcl = pytorch3d.structures.Pointclouds(points=[points], features=[color],).to(device)
        render360GIFfromPCL(taurus_pcl,'Torus',output_path,args.image_size,viz=True)

        #Spherical Spiral https://math.stackexchange.com/questions/3574112/parameterization-of-a-spherical-spiral#:~:text=For%20a%20spherical%20spiral%20curve,%CF%80%5D%20and%20c%20a%20constant.
        c, r = 1, 2
        t = torch.linspace(0, np.pi, args.num_samples)
        x = r * torch.sin(t) * torch.cos(c*t)
        y = r * torch.sin(t) * torch.sin(c*t)
        z = r * torch.cos(t)

        points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
        color = (points - points.min()) / (points.max() - points.min())

        ss_pcl = pytorch3d.structures.Pointclouds(points=[points], features=[color],).to(device)
        render360GIFfromPCL(ss_pcl,'Spherical Spiral',output_path,args.image_size,viz=True)

    elif args.q == 5.3:
        voxel_size = 100
        min_value = -1.1
        max_value = 1.1
        X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
        
        #TORUS
        r, R = 0.45, 0.55
        voxels = (R - torch.sqrt(X**2 + Y**2) )**2 + Z**2 - r**2

        vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
        vertices = torch.tensor(vertices).float()
        faces = torch.tensor(faces.astype(int))
        # Vertex coordinates are indexed by array position, so we need to
        # renormalize the coordinate system.
        vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
        textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
        textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

        mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures)
        mesh = mesh.to(device)
        render360GIFfromMesh(mesh,'torus_implicit',output_path,args.image_size,viz=True)
        

        #CONE
        h, r = 3, 0.3
        voxels = (X**2 + Y**2) / r**2 - ((Z - h)**2) / h**2

        vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
        vertices = torch.tensor(vertices).float()
        faces = torch.tensor(faces.astype(int))
        # Vertex coordinates are indexed by array position, so we need to
        # renormalize the coordinate system.
        vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
        textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
        textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

        mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures)
        mesh = mesh.to(device)
        render360GIFfromMesh(mesh,'cone_implicit',output_path,args.image_size,viz=True)

    elif args.q == 6:

        # Define the parametric function of the helix on a cone
        def helix_on_cone(t, height_factor=0.5, frequency=8):
            r = 0.03 - 0.01 * t
            x = (1 - t) * torch.cos(frequency * t)
            y = (1 - t) * torch.sin(frequency * t)
            z = height_factor * t  # Height factor reduces the height of the cone
            return r * x, r * y, z

        # Create a grid of t values
        t_values = torch.linspace(0, 6 * np.pi, 1000)

        # Calculate the surface points
        x, y, z = helix_on_cone(t_values)

        points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
        
        colors = (points - points.min()) / (points.max() - points.min())
        light_grey = torch.tensor([63, 63, 63]) / 255
        dark_grey = torch.tensor([192, 192, 192]) / 255
        color_range = light_grey * (1 - colors) + dark_grey * colors

        pcl = pytorch3d.structures.Pointclouds(points=[points], features=[color_range],).to(device)

        render360imageslist = []
        renderer = utils.get_points_renderer(image_size=args.image_size, device=device, background_color=(1, 1, 1))

        for i in range(args.num_frames):
            # R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=20, elev=0, azim=i, degrees=True)
            R = pytorch3d.transforms.euler_angles_to_matrix(torch.tensor([np.pi/2, 0, 0]), "XYZ").float().unsqueeze(0).to(device)
            T = (torch.tensor([0, -5, 10]).float().unsqueeze(0) + torch.rand(1, 3)).to(device)
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

            rend = renderer(pcl, cameras=cameras)
            image = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)

            #convert the image to uint8 for compatibility with plt and imageio
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
            if True:
            #visualize the images
                plt.imshow(image)
                plt.title('Whirlpool')
                plt.show(block=False)
                plt.pause(0.05)

            render360imageslist.append(image)

        print('Saving the 360 GIF for ' + 'whirlpool' + ' in ' + output_path)
        imageio.mimsave(os.path.join(output_path,'whirpool'+'.gif'), render360imageslist, fps=15)

    elif args.q == 7:
        raise('Not implemented')

    else:
        raise('Invalid question value')



