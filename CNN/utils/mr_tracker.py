import numpy as np
from sklearn.svm import SVC
from scipy.ndimage.measurements import center_of_mass
from skimage import measure
from skimage import transform
from skimage.color import gray2rgb
from skimage.exposure import rescale_intensity
from skimage.measure import find_contours

from itertools import product

from cardiac_segmentation.dataset.processing import pad_data
from cardiac_segmentation.dataset.processing import crop_data

from .vis import histogram_equalize

def fit_line(points):
    """
    Fit a line with points given w/ least squares
        Line is y = {m}x + {c}
        return [m, -1], c
    """
    x_coords, y_coords = np.array(points).T
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]
    return np.array([m, -1]), c


def line_separation(pts0, pts1):
    """
    Fit a line for separating the two groups of points
        return (w, b)
    """
    train_x = np.concatenate([pts0, pts1])
    train_y = np.concatenate([np.zeros(pts0.shape[0]), np.ones(pts1.shape[0])])
    clf = SVC(kernel='linear')
    clf.fit(train_x, train_y)
    w = clf.coef_[0]
    b = clf.intercept_[0]
    return w, b


def contour_to_valve_pts(contour, valve_line, center_line, margin=1):
    """
    Get the two points that describe the connection of the valve to the heart
        in the segmentation mask
        - contour (Left atrium contour or left ventricle contour
        - valve_line (line separating LA mask and LV mask
        - center_line (line connecting the center of LA and LV)
    """

    margins = contour @ valve_line[0] + valve_line[1]
    margin_index = np.where(np.abs(margins)<margin)[0]
    margin_pts = contour[margin_index]

    offsets = margin_pts @ center_line[0] + center_line[1]
    valve_pts_index = margin_index[np.array([np.argmin(offsets), np.argmax(offsets)])]
    valve_pts = contour[valve_pts_index]

    return valve_pts

def find_valve_pts(mask, margin=1):
    """
    Find the valve pts given the segmentation masks
        - la_mask label 4
        - lv_mask label 1
    """
    la_mask = mask==4
    lv_mask = mask==1

    # get the line separating LA and LV
    lv_pts = np.vstack(np.where(lv_mask)).T
    la_pts = np.vstack(np.where(la_mask)).T

    valve_line = line_separation(la_pts, lv_pts)

    # get the line connecting LA and LV centers
    la_cc = center_of_mass(la_mask)
    lv_cc = center_of_mass(lv_mask)

    center_line = fit_line([la_cc, lv_cc])

    # get contour pts
    lv_contour = measure.find_contours(lv_mask, 0.5)[0]
    la_contour = measure.find_contours(la_mask, 0.5)[0]

    lv_valve_pts = contour_to_valve_pts(lv_contour, valve_line, center_line, margin)
    la_valve_pts = contour_to_valve_pts(la_contour, valve_line, center_line, margin)

    return lv_valve_pts, la_valve_pts, lv_contour, la_contour, valve_line, center_line



def get_valve_patch(mask, image,
                    la_margin=10,
                    lv_margin=5,
                    valve_margin=2,
                    boundary_guard=True):
    """
    Rotate the image according to the valve orientation, then cut a patch around the valve
        - la_margin: 10 pixels into left atrium
        - lv_maring: 5 pixels into the left ventricle
        - valve_margin: 2 pixels into the valve wall (myocardium)
    """

    lv_valve_pts, la_valve_pts, lv_contour, la_contour, valve_line, center_line = find_valve_pts(mask, margin=2)

    w = valve_line[0]
    cc = np.concatenate([la_valve_pts, lv_valve_pts]).mean(axis=0)

    rot = transform.EuclideanTransform(
        rotation=np.arctan(w[1]/w[0]),
        )
    t_cc = rot(cc[::-1])[0]

    img_height, img_width = image.shape[-2:]
    t_c = img_width/2 - t_cc[0]
    t_r = img_height/2 - t_cc[1]

    tform = transform.EuclideanTransform(
        rotation=np.arctan(w[1]/w[0]),
        #translation=(-104,81),
        translation=(t_c,t_r),
        )

    if boundary_guard:
        row_min = int(tform(la_contour[:,::-1])[:,1].min().round())
        row_max = int(tform(lv_contour[:,::-1])[:,1].max().round())
        col_min_ck = min(int(tform(la_contour[:,::-1])[:,0].min().round()),
                         int(tform(lv_contour[:,::-1])[:,0].min().round()))
        col_max_ck = max(int(tform(la_contour[:,::-1])[:,0].max().round()),
                         int(tform(lv_contour[:,::-1])[:,0].max().round()))
    else:
        row_min = 0
        row_max = img_height-1
        col_min_ck = 0
        col_max_ck = img_width-1

    tform_valve_pts = tform(la_valve_pts[:,::-1])
    row = int(tform_valve_pts[:,1].mean().round())
    col_min = int(tform_valve_pts[:,0].min().round())
    col_max = int(tform_valve_pts[:,0].max().round())


    row_min = max(row - la_margin, row_min)
    row_max = min(row + lv_margin, row_max)

    col_min = max(col_min_ck, col_min - valve_margin)
    col_max = min(col_max_ck, col_max + valve_margin)

    tf_img = transform.warp(image, tform.inverse)
    return tf_img[row_min:row_max, col_min:col_max], tform, ((row_min, row_max), (col_min, col_max))



def get_patch_mask(shape, tform, idx):
    """
    Get the patch mask in the original orientation (tform)
    """
    row_idx, col_idx = idx
    patch_mask = np.zeros(shape)
    patch_mask[row_idx[0], col_idx[0]:col_idx[1]] = 1
    patch_mask[row_idx[1], col_idx[0]:col_idx[1]] = 1
    patch_mask[row_idx[0]:row_idx[1], col_idx[0]] = 1
    patch_mask[row_idx[0]:row_idx[1], col_idx[1]] = 1

    patch_mask_filled = np.zeros(shape)
    patch_mask_filled[row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]] = 1

    patch_mask_og = transform.warp(patch_mask, tform)
    patch_mask_filled_og = transform.warp(patch_mask_filled, tform)

    return patch_mask_og, patch_mask_filled_og


def get_og_patch_corner_coords(tform, row_idx, col_idx):
    coords = np.array([(row_idx[0], col_idx[0]), 
                       (row_idx[0], col_idx[1]),
                       (row_idx[1], col_idx[1]),
                       (row_idx[1], col_idx[0])])

    og_coords = tform.inverse(coords[:,::-1])
    return og_coords


def get_patch_stack_anchor(images, masks, n_frames=15, anchor=0, image_size=(32,32),
                           margins=(10,5,2), verbose=False, reshape=True, boundary_guard=True):
    """
    Use the anchor frame to create the patch mask/location for all frames
    """
    mask_index = 4

    patch, tform, (row_idx, col_idx) = get_valve_patch(masks[anchor], images[anchor], *margins, 
                                                       boundary_guard=boundary_guard)

    padding = pad_data(image_size)
    cropping = crop_data(image_size, static=True)
    patches = []
    for i in range(n_frames):
        tf_img = transform.warp(images[i], tform.inverse)
        patch = tf_img[row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]]
        if reshape:
            patches.append(cropping(padding(patch)))
        else:
            patches.append(patch)

    patches = np.array(patches)

    if verbose:
        imgs_outlined = []
        imgs_outlined2 = []
        patch_masks = []

        patch_mask, patch_mask_filled = get_patch_mask(images[anchor].shape[:2], tform, (row_idx, col_idx))
        coords = get_og_patch_corner_coords(tform, row_idx, col_idx)
        for i in range(n_frames):
            img_rgb = gray2rgb(rescale_intensity(images[i], out_range=np.uint8))
            img_rgb[np.where(patch_mask)] = (0, 255, 0)
            imgs_outlined.append(img_rgb)

            contours = find_contours(masks[i]==mask_index, 0.5)[0].astype(int)
            img_rgb2 = img_rgb.copy()
            img_rgb2[(contours[:,0], contours[:,1])] = (255, 0, 0)
            imgs_outlined2.append(img_rgb2)

            patch_masks.append(patch_mask_filled)

        imgs_outlined = np.array(imgs_outlined)
        imgs_outlined2 = np.array(imgs_outlined2)
        patch_masks = np.array(patch_masks)
        return patches, patch_masks, imgs_outlined, imgs_outlined2, coords
    else:
        return patches


def get_patch_stack_fluid(images, masks, n_frames=15, image_size=(32,32),
                          margins=(10,5,2), verbose=False, reshape=True,
                          boundary_guard=True):
    """
    Use the corresponding frame to create the patch mask/location for all frames
    """
    mask_index = 4

    if verbose:
        imgs_outlined = []
        imgs_outlined2 = []

    padding = pad_data(image_size)
    cropping = crop_data(image_size, static=True)
    patches = []
    patch_masks = []
    corner_coords = []
    for i in range(n_frames):
        patch, tform, (row_idx, col_idx) = get_valve_patch(masks[i], images[i], *margins,
                                                           boundary_guard=boundary_guard)
        if reshape:
            patches.append(cropping(padding(patch)))
        else:
            patches.append(patch)

        if verbose:
            patch_mask, patch_mask_filled = get_patch_mask(images[i].shape[:2], tform, (row_idx, col_idx))
            img_rgb = gray2rgb(rescale_intensity(images[i], out_range=np.uint8))
            img_rgb[np.where(patch_mask)] = (0, 0, 255)
            imgs_outlined.append(img_rgb)

            contours = find_contours(masks[i]==mask_index, 0.5)[0].astype(int)
            img_rgb2 = img_rgb.copy()
            img_rgb2[(contours[:,0], contours[:,1])] = (255, 0, 0)
            imgs_outlined2.append(img_rgb2)

            patch_masks.append(patch_mask_filled)

            coords = get_og_patch_corner_coords(tform, row_idx, col_idx)
            corner_coords.append(coords)

    patches = np.array(patches)
    patch_masks = np.array(patch_masks)

    if verbose:
        imgs_outlined = np.array(imgs_outlined)
        imgs_outlined2 = np.array(imgs_outlined2)
        corner_coords = np.array(corner_coords)

        return patches, patch_masks, imgs_outlined, imgs_outlined2, corner_coords
    else:
        return patches




def get_patch_mapping_hist(pid, **kwargs):

    load_image = kwargs.get("load_image")
    load_mask = kwargs.get("load_mask")
    method = kwargs.get("method", 0)
    boundary_guard = kwargs.get("boundary_guard", True)
    margins = kwargs.get("margins", (15,15,4))
    image_size = kwargs.get("image_size", (32,32))
    n_frames = kwargs.get("n_frames", 50)

    images = load_image(pid)
    masks = load_mask(pid)

    if method == 0:
        get_patch_stack_method = get_patch_stack_anchor
    else:
        get_patch_stack_method = get_patch_stack_fluid

    patch_results = \
                get_patch_stack_method(images, masks,
                                       n_frames=n_frames, anchor=0,
                                       verbose=True, reshape=True,
                                       image_size=image_size,
                                       margins=margins,
                                       boundary_guard=boundary_guard)
    patches, patch_masks, imgs_outlined, imgs_outlined2 = patch_results[:4]


    la_mapping = histogram_equalize(images, masks==4, None)
    patch_mapping = histogram_equalize(images, patch_masks.round().astype(bool), None)
    la_mapped_images = la_mapping(images).astype(float)
    patch_mapped_images = patch_mapping(images).astype(float)

    la_patch_results = \
                get_patch_stack_method(la_mapped_images, masks,
                                       n_frames=n_frames, anchor=0,
                                       verbose=True, reshape=True,
                                       image_size=image_size,
                                       margins=margins,
                                       boundary_guard=boundary_guard)
    la_patches, la_patch_masks, la_imgs_outlined, la_imgs_outlined2 = la_patch_results[:4]

    pa_patch_results = \
                get_patch_stack_method(patch_mapped_images, masks,
                                       n_frames=n_frames, anchor=0,
                                       verbose=True, reshape=True,
                                       image_size=image_size,
                                       margins=margins,
                                       boundary_guard=boundary_guard)
    pa_patches, pa_patch_masks, pa_imgs_outlined, pa_imgs_outlined2 = pa_patch_results[:4]

    original_results = (patches, patch_masks, imgs_outlined, imgs_outlined2)
    la_mapped_results = (la_patches, la_patch_masks, la_imgs_outlined, la_imgs_outlined2)
    pa_mapped_results = (pa_patches, pa_patch_masks, pa_imgs_outlined, pa_imgs_outlined2)

    return original_results, la_mapped_results, pa_mapped_results, patch_results[-1]


