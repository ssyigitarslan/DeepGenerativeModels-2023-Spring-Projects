import numpy as np
from PIL import Image
import torch
from patches import aggregate_patches, get_image_patches
from nn import nearest_neighbors_pytorch, nearest_neighbors

def pnn(image, Q, K, V, V_shape, alpha, P, C, device):
    if device == 'cuda':
      nn = nearest_neighbors_pytorch(Q, K, alpha, device) # GPU
    else:
      nn = nearest_neighbors(Q, K, alpha) # CPU
    output = V[nn] # Select V indexes
    output = output.reshape(V_shape[0], V_shape[1], P, P, C)
    output = aggregate_patches(image, output, P)
    
    return output

def gpnn(image, N, T, R, alpha, P, C, device):
    # Pyramid of smaller images, descending in size
    smaller_image_arrs = [np.asarray(image.resize((int(image.size[0]/(i * R)), int(image.size[1]/(i * R)))), dtype=np.uint16) for i in range(1, N+1)]
    smaller_image_arrs.insert(0, np.asarray(image))

    for n in range(N): # Should be N+1, but restricted to N due to memory issues
        if not n: # T=1 for Nth step
            #Nth step (coarsest scale)
            noise = np.random.normal(size=smaller_image_arrs[-1].shape, scale=0.75)
            Q = smaller_image_arrs[-1] + noise # y_n+1
            Q = get_image_patches(Q, P)
            Q = Q.reshape(-1, P, P, C)

            V = smaller_image_arrs[-1] # x_n
            V = get_image_patches(V, P)
            V_shape = V.shape
            V = V.reshape(-1, P, P, C)

            K = smaller_image_arrs[-1] # x_n+1
            K = get_image_patches(K, P)
            K = K.reshape(-1, P, P, C)

            generated = pnn(smaller_image_arrs[-1], Q, K, V, V_shape, alpha, P, C, device)

        else:
            # Generation from the prev step is resized for the next step
            generated = Image.fromarray(generated.astype(np.uint8)).resize((smaller_image_arrs[-1 * (1 + n)].shape[0], smaller_image_arrs[-1 * (1 + n)].shape[1]))
            generated = np.asarray(generated, dtype=np.uint16)

            # V is the source image at n
            V = smaller_image_arrs[-1 * (1 + n)] # x_n
            V = get_image_patches(V, P)
            V_shape = V.shape
            V = V.reshape(-1, P, P, C)

            # K is the source image at n+1, upscaled (blurry)
            K = Image.fromarray(smaller_image_arrs[-1 * (n)].astype(np.uint8)).resize((smaller_image_arrs[-1 * (1 + n)].shape[1], smaller_image_arrs[-1 * (1 + n)].shape[0])) # Width and height get reversed here
            K = np.asarray(K, dtype=np.uint16)
            K = get_image_patches(K, P)
            K = K.reshape(-1, P, P, C)

            # Q is the initial guess, at first from the previous step and
            # then updated at every t as the current guess at step n
            for t in range(T):
                Q = generated
                Q = get_image_patches(Q, P)
                Q = Q.reshape(-1, P, P, C)

                generated = pnn(smaller_image_arrs[-1 * (1 + n)], Q, K, V, V_shape, alpha, P, C, device)

        Image.fromarray(generated.astype(np.uint8)).show() # Enable this to show pyramid generations
        torch.cuda.empty_cache()

    return Image.fromarray(generated.astype(np.uint8))


