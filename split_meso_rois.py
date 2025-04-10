import os
from pathlib import Path
from pprint import pprint
import tifffile
import json
import re
import numpy as np
from tifffile.tifffile import COMPRESSION
from tifffile import read_scanimage_metadata
import pickle

def split_tiff_to_roi_streamed(tiff_path, roi_divisions=None, chunk_size=100, delete_raw_tif=False):
    """
    Splits a multi-frame TIFF into ROI vertical segments and saves each segment as a
    streamed multi-page BigTIFF using the same compression as the source. Outputs are
    ImageJ-compatible and written into RXXX/ subfolders.

    Parameters:
        tiff_path (str or Path): Path to the input TIFF file.
        roi_divisions (list[list[int, int]] or None): List of [start_row, end_row] pixel indices for each ROI.
                                                      If None or empty, the full frame is used as a single ROI.
        chunk_size (int): Number of frames to process at once.
        delete_raw_tif (bool): If True, deletes the original TIFF after processing.
    """
    if not chunk_size or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer greater than 0")

    tiff_path = Path(tiff_path)
    tiff_dir = tiff_path.parent
    tiff_name = tiff_path.stem

    with tifffile.TiffFile(tiff_path) as tif:
        num_frames = len(tif.pages)
        frame_height, frame_width = tif.pages[0].shape

        # Default: whole frame as one ROI
        if not roi_divisions:
            roi_divisions = [[0, frame_height]]

        # Validate all ranges
        for pair in roi_divisions:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                raise ValueError("Each ROI division must be a [start, end] pair.")
            if pair[0] < 0 or pair[1] > frame_height or pair[0] >= pair[1]:
                raise ValueError(f"Invalid ROI range: {pair}")

        num_rois = len(roi_divisions)

        # Get source compression
        compression_tag = tif.pages[0].tags.get('Compression')
        compression = compression_tag.value if compression_tag else None
        if isinstance(compression, int):
            compression = COMPRESSION(compression).name.lower()

        # Get ImageJ-style description
        image_description = tif.pages[0].tags.get('ImageDescription')
        description = image_description.value if image_description else ""

        # Setup output writers
        roi_writers = []
        for roi_idx in range(num_rois):
            roi_name = f"R{roi_idx+1:03d}"
            roi_folder = tiff_dir / roi_name
            roi_folder.mkdir(exist_ok=True)
            out_path = roi_folder / f"{tiff_name}_{roi_name}_full.tif"

            writer = tifffile.TiffWriter(out_path, bigtiff=True)
            roi_writers.append(writer)

        try:
            # Process in chunks
            for start in range(0, num_frames, chunk_size):
                end = min(start + chunk_size, num_frames)
                # print(f"Processing frames {start} to {end - 1}")
                chunk = tif.asarray(key=range(start, end))  # (chunk_size, height, width)
                if chunk.ndim == 2:
                    chunk = np.expand_dims(chunk, axis=0)

                for frame in chunk:
                    for roi_idx, (top, bottom) in enumerate(roi_divisions):
                        segment = frame[top:bottom, :]
                        roi_writers[roi_idx].write(
                            segment,
                            photometric='minisblack',
                            metadata={'ImageJ': True},
                            description=description,
                            compression=compression
                        )
        finally:
            for writer in roi_writers:
                writer.close()

    if delete_raw_tif:
        try:
            print(f"Deleting original TIFF: {tiff_path}")
            tiff_path.unlink()
        except Exception as e:
            print(f"Warning: Failed to delete original TIFF: {e}")

    print(f"✅ Done. Saved {num_rois} ROI TIFFs from {num_frames} frames.")

def calculate_roi_ranges_from_heights(roi_heights, spacer_pixels=0):
    """
    Converts a list of ROI heights and a spacer size into
    a list of [start_row, end_row] pairs for each ROI.

    Args:
        roi_heights (list[int]): Height in pixels of each ROI.
        spacer_pixels (int): Number of pixels between ROIs.

    Returns:
        list[list[int, int]]: ROI Y-coordinate start/end rows.
    """
    roi_ranges = []
    current_start = 0

    for height in roi_heights:
        start = current_start
        end = start + height
        roi_ranges.append([start, end])
        current_start = end + spacer_pixels

    return roi_ranges

def extract_full_tiff_metadata(tiff_path):
    metadata = {}

    with tifffile.TiffFile(tiff_path) as tif:
        page = tif.pages[0]

        # Standard IFD tags
        for tag in page.tags.values():
            name = tag.name
            value = tag.value
            metadata[name] = value

        # ImageJ metadata block
        if hasattr(page, 'imagej_metadata') and page.imagej_metadata:
            metadata['ImageJ_Metadata'] = page.imagej_metadata

        # ScanImage-style Artist tag parsing (includes ROI data)
        artist_tag = page.tags.get('Artist')
        if artist_tag:
            artist_text = artist_tag.value.strip()

            # Clean formatting for JSON parsing
            artist_text = re.sub(r',\s*([}\]])', r'\1', artist_text)  # remove trailing commas
            artist_text = artist_text.replace("null", "null")
            artist_text = artist_text.replace("NaN", "null")

            try:
                artist_parsed = json.loads(artist_text)
                metadata['Artist_Parsed'] = artist_parsed
            except json.JSONDecodeError as e:
                metadata['Artist_Parsed'] = f"Failed to parse Artist tag: {e}"
        else:
            metadata['Artist_Parsed'] = None

    return metadata

def split_meso_rois(exp_dir_raw='debug',delete_raw_tifs=False):
        if exp_dir_raw == 'debug':
            exp_dir_raw = '/home/adamranson/data/tif_meso/1'

        # make list of all scan paths
        scanpath_names = []
        # cycle through checking if folders exist for each scan path from 0 to 9
        for i in range(10):
            # check if folder exists with name P + i
            path = os.path.join(exp_dir_raw, 'P' + str(i))
            if os.path.exists(path):
                # if it exists add to list
                scanpath_names.append(path)
                # cycle through tifs within the folder of the path splitting them at the roi divisions
                # pixel positions where the roi divisions are - will later be read from meta data, 
                # and be specific to each scan path
                roi_divisions = []
                # detect all tif file names within the path
                all_tif_files_in_path = sorted(
                    [f for f in os.listdir(path) if f.lower().endswith('.tif')],
                    key=lambda x: int(re.search(r'\d+', x).group())
                )
                # cycle through all tif files in the path
                metadata_dict = extract_full_tiff_metadata(os.path.join(path,all_tif_files_in_path[0]))
                # read scanimage meta data
                with open(os.path.join(path,all_tif_files_in_path[0]), 'rb') as tif_meta_reader: 
                    si_all_meta = read_scanimage_metadata(tif_meta_reader)
                    # pprint(si_all_meta)
                number_slices = si_all_meta[0]['SI.hStackManager.numSlices']
                # Compile meta data for saving
                SI_meta = {}
                SI_meta['Meta1'] = si_all_meta
                SI_meta['Meta2'] = metadata_dict
                # Access ROI metadata 
                rois = metadata_dict.get("Artist_Parsed", {}).get("RoiGroups", {}).get("imagingRoiGroup", {}).get("rois", [])
                print(f"\nFound {len(rois)} ROIs:")
                for i, roi in enumerate(rois):
                    print(f"  ROI {i+1}: {roi.get('name')}")
                    if roi.get('enable')==1:
                        # Get the scanfields section (can be a list or single dict)
                        scanfields = roi.get("scanfields")
                        # Handle if scanfields is a list or a single dict
                        if isinstance(scanfields, list):
                            # use first scan field to get pixel resolution
                            pixres = scanfields[0]['pixelResolutionXY']
                            roi_divisions.append(pixres[1])
                            for j, sf in enumerate(scanfields):
                                pixres = sf.get("pixelResolutionXY")
                                enabled = sf.get("enable")
                                # print(f"ROI {i+1} - Scanfield {j+1}: pixelResolutionXY = {pixres}")
                                # print(f"ROI {i+1} - Scanfield {j+1}: enabled = {enabled}")

                        elif isinstance(scanfields, dict):
                            pixres = scanfields.get("pixelResolutionXY")
                            roi_divisions.append(pixres[1])
                            # print(f"ROI {i+1}: pixelResolutionXY = {pixres}")
                        else:
                            print(f"ROI {i+1}: No scanfields found.")

                # calculate the lines to discard between the roi divisions
                spacer_pixels = (metadata_dict['ImageLength']-sum(roi_divisions))/(len(roi_divisions)-1)
                if round(spacer_pixels) != spacer_pixels:
                    raise ValueError("The spacer pixels are not an integer. Check the metadata.")
                else:
                    spacer_pixels = int(spacer_pixels)
                # print('Calculated spacer pixels:',spacer_pixels)
                # calculate the roi divisions with the spacer pixels taken into account
                # for testing:
                # roi_divisions = [200,200,100]
                # spacer_pixels = 6
                roi_ranges = calculate_roi_ranges_from_heights(roi_divisions, spacer_pixels=spacer_pixels)             

                for tif_file in all_tif_files_in_path:
                    full_tif_path = os.path.join(path,tif_file)
                    split_tiff_to_roi_streamed(full_tif_path, roi_divisions=roi_ranges, chunk_size=100, delete_raw_tif=delete_raw_tifs)

                # dump SI_meta to a np file
                with open(os.path.join(path, 'SI_meta.pickle'), 'wb') as f: pickle.dump(SI_meta, f)

def check_and_process_experiments(base_dir):
    for animal_id in os.listdir(base_dir):
        animal_path = os.path.join(base_dir, animal_id)
        if not os.path.isdir(animal_path):
            continue
        
        for exp_id in os.listdir(animal_path):
            exp_path = os.path.join(animal_path, exp_id)
            if not os.path.isdir(exp_path):
                continue

            needs_processing = False
            for scanpath in ['P1', 'P2']:
                scanpath_path = os.path.join(exp_path, scanpath)
                if os.path.isdir(scanpath_path):
                    meta_path = os.path.join(scanpath_path, 'SI_meta.pickle')
                    if not os.path.isfile(meta_path):
                        needs_processing = True
                        break  # No need to keep checking if already missing
                else:
                    # Folder doesn't exist, skip or log if needed
                    continue

            if needs_processing:
                print('')
                print(f"Processing {exp_path}...")
                try:
                    split_meso_rois(exp_path, delete_raw_tifs=False)
                except Exception as e:
                    print(f"Error processing {exp_path}: {e}")
            else:
                print(f"Skipping {exp_path}, already processed.")

# for debugging:
def main():
    local_repo_directory = '/home/adamranson/data/tif_meso/local_repository'
    check_and_process_experiments(local_repo_directory)
    
if __name__ == "__main__":
    main()