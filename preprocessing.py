import pandas as pd
import torch
from pathlib import Path
import numpy as np
import skimage
import scipy
import pydicom as dicom

def crop_to_mask(image: np.ndarray, mask: np.ndarray, padding=20):
    """
    Crop the mammography image and the segmentation mask keeping only the lesion pixels, with some additional padding contour.

    :param image: the original mammography image
    :param mask: the binary mask image
    :param padding: additional padding to add to the contour of the roi, in order to acquire some of the lesion's context
    """
    # Find nonzero coordinates
    coords = np.argwhere(mask)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Add padding
    y_min = max(y_min - padding, 0)
    x_min = max(x_min - padding, 0)

    y_max = min(y_max + padding, image.shape[0])
    x_max = min(x_max + padding, image.shape[1])

    # Crop
    cropped_image = image[y_min:y_max, x_min:x_max]
    cropped_mask  = mask[y_min:y_max, x_min:x_max]

    return cropped_image, cropped_mask

def preprocess_image(image: np.ndarray, mask: np.ndarray, padding: int = 20) -> np.ndarray:
    """
    Apply preprocessing operations to the mammography image, by performing median filtering, clahe and then unsharp.
    Finally, crop the image keeping only the roi pixels, with some additional padding contour, and downscale the image
    to 512x512

    :param image: the original mammography image
    :param mask: the binary mask image
    :param padding: additional padding to add to the contour of the roi, in order to acquire some of the lesion's context
    """
    image_target_shape = (512, 512)

    # application of median filtering
    median_filtered_image = scipy.ndimage.median_filter(image, size=3)

    # application of clahe
    clahe_image = skimage.exposure.equalize_adapthist(median_filtered_image)

    # application of unsharp
    preproc_image = skimage.filters.unsharp_mask(clahe_image)

    # crop around the roi
    preproc_image, _ = crop_to_mask(preproc_image, mask, padding)

    # downscale the image
    preproc_image = skimage.transform.resize(preproc_image, image_target_shape)

    return preproc_image

def preprocess_mask(image: np.ndarray, mask: np.ndarray, padding: int = 20) -> np.ndarray:
    """
    Apply preprocessing to the segmentation mask, by cropping the image keeping only the roi pixels, with some additional
    padding contour, and downscale the image to 512x512

    :param image: the original mammography image
    :param mask: the binary mask image
    :param padding: additional padding to add to the contour of the roi, in order to acquire some of the lesion's context
    """
    mask_target_shape = (512, 512)

    # crop around the roi
    _, preproc_mask = crop_to_mask(image, mask, padding)

    # downscale the image
    preproc_mask = skimage.transform.resize(preproc_mask, mask_target_shape)

    return preproc_mask

def preprocess_dataset(original_df: pd.DataFrame, data_root_filepath: str) -> pd.DataFrame:
    """
    Apply preprocessing to both the whole dataset of images and segmentation masks. Median filtering, clahe and then
    unsharp are applied to the images. Then, both the images and the masks are cropped keeping only the roi pixels, with
    some additional padding contour, and are downscaled to 512x512. Finally, a folder called "preprocessed_lesions" is
    created in the main data folder, and both the images and the lesions are exported there as float32 pytorch tensors.

    :param original_df: dataset containing the metadata information of the CBIS-DDSM dataset
    :param data_root_filepath: filepath to the main data folder
    """
    df = original_df.copy(deep=True)

    # Specify the target directory path for the preprocessed data
    Path(f"{data_root_filepath}/preprocessed_lesions").mkdir(parents=True, exist_ok=True)

    # lists containing the filepaths for preprocessed lesions and masks
    preproc_image_filepaths = []
    preproc_masks_filepaths = []

    # iterate over all patients
    last_patient = ""
    roi_index = 0
    for _, row in df.iterrows():
        subject_id = row["Subject ID"]
        patient_folder_name = f"{subject_id}"

        if subject_id != last_patient:
            last_patient = subject_id
            roi_index = 0
        else:
            roi_index += 1

        fullimage_filepath = f"{data_root_filepath}/" + row["fullimage filepath"]
        roi_mask_filepath = f"{data_root_filepath}/" + row["roi filepath"]
        preproc_fullimage_filepath = f"{data_root_filepath}/preprocessed_lesions/{patient_folder_name}/fullimage_{roi_index}.pt"
        preproc_mask_filepath = f"{data_root_filepath}/preprocessed_lesions/{patient_folder_name}/roi_{roi_index}.pt"

        # load up the dicom files
        image = dicom.dcmread(fullimage_filepath).pixel_array
        mask = dicom.dcmread(roi_mask_filepath).pixel_array

        # if folder does not exist, create folder for the patient preprocessed data
        Path(f"{data_root_filepath}/preprocessed_lesions/{patient_folder_name}").mkdir(parents=True, exist_ok=True)

        # check if the preprocessed mammography already exists, otherwise create it
        if not Path(preproc_fullimage_filepath).is_file():
            # preprocess the lesion
            preproc_fullimage = preprocess_image(image, mask)

            # cast the image to tensor
            preproc_fullimage_tensor = torch.as_tensor(preproc_fullimage.astype(np.float32)).unsqueeze(0)

            # save up the image tensor
            with open(preproc_fullimage_filepath, "wb") as f:
                torch.save(preproc_fullimage_tensor, f)

        # check if the preprocessed roi already exists, otherwise create it
        if not Path(preproc_mask_filepath).is_file():
            # preprocess the mask
            preproc_mask = preprocess_mask(image, mask)

            # Create one-hot encoded channels
            channel_0 = (preproc_mask == 0).astype(np.float32)
            channel_1 = (preproc_mask != 0).astype(np.float32)
            stacked_array = np.stack([channel_0, channel_1], axis=-1)

            preproc_tensor_mask = torch.from_numpy(np.transpose(stacked_array, (2, 0, 1)))

            # save up the tensor mask
            with open(preproc_mask_filepath, "wb") as f:
                torch.save(preproc_tensor_mask, f)

        # save up the filepaths
        preproc_image_filepaths.append(f"preprocessed_lesions/{patient_folder_name}/fullimage_{roi_index}.pt")
        preproc_masks_filepaths.append(f"preprocessed_lesions/{patient_folder_name}/roi_{roi_index}.pt")

    df["preprocessed fullimage tensor filepath"] = preproc_image_filepaths
    df["preprocessed mask tensor filepath"] = preproc_masks_filepaths

    return df