# Gnazim Project projectid: gnazim-project

import os
import cv2
import pandas as pd
from ParagraphDetector import ImageProcessor
import re
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import time
import functools
from datetime import datetime
from typing import Callable, Any

# This dict will store the function name as the key and a list [total_time, call_count, run_timestamp] as the value.
profile_data = {}




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pseudo Code Logic:
# Initialize GCP Connection
# Initialize Queue with Root Folder ID
# Set Reconnection Threshold for GCP
# Set File Processing Count Threshold
# Set Queue Save and Read Threshold

# While Queue is not empty:
#    Dequeue a Folder from Queue
#    Retrieve Data of Previously Processed and Problematic Files from Local Storage
#    Count Already Processed Files and Create Set of Processed File Paths
#    Initialize Counter for Files Processed in Current Run

#    If Reconnection Threshold is Reached:
#        Reconnect to GCP

#    If File Processing Count Threshold is Reached:
#        Save Current State of Queue to Local Storage

#    Get List of Files in Current Folder
#    Initialize DataFrame for New Processed Files

#    For Each File in Folder:
#        If File is an Image and Not Already Processed:
#            Extract Meta Data from File Path (gcp_extract_years_author_type)
#            Download File from GCP
#            Process Image to Detect Relevant Paragraphs (ImageProcessor)
#            Extract Text from Detected Paragraphs using GCP OCR (gcp_extract_text_from_image)
#            Append Processed File Data to New DataFrame

#            If GCP Connection Error Occurs:
#                Attempt Reconnection (gcp_reconnect) and Reprocess File
#                If Reconnection Fails:
#                    Log Error and Update Problematic Files Data in Local Storage
#                   Break from Loop

#        Else If File is Not an Image:
#            Add File (Folder) to Queue for Later Processing

#     If Folder is Successfully Processed:
#         Append New Data to DataFrame of Processed Files
#         Update Local CSV Files with New Processed and Problematic Files Data

#     If Queue Save and Read Threshold is Reached:
#         Read Next Set of Folders from Saved Queue State

#     Perform Function Profiling for Each Function Call

# After Queue is Empty:
#     Generate GCP File and Folder Links using create_df_gcp_file_links
#     Append Links to Processed Files DataFrame
#     Save Updated DataFrame to Local Storage

# Update Profiling Data with Run Timestamp at End of Run
# Perform Overall Time Analysis*

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ************************************************************************************************************************

DATA_COL_NAMES = ['identifier', 'path', 'gcp_file_id', 'folder_name', 'author_subject', 'type',
                  "Years",
                  'gcp_folder_id', 'file_name', 'ocr_writen_on', 'ocr_writen_by',
                  'ocr_main_content', 'ocr_additional_content',

                  'ocr_writen_on_coords', 'ocr_writen_by_coords', 'ocr_main_content_coords',
                  'ocr_additional_content_coords',

                  'paragraphs_detection_successes', 'ocr_all_text_preprocess', 'ocr_all_text_no_preprocess',
                  'ocr_main_content_all_text_preprocess', 'ocr_main_content_all_text_no_preprocess',"years"]

PROBLEM_COL_NAMES = DATA_COL_NAMES + [ "error_message"]


FILE_EXTENSIONS = [
    '.txt', '.doc', '.docx', '.pdf', '.rtf',
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff',
    '.mp3', '.wav', '.ogg', '.flac',
    '.mp4', '.avi', '.mov', '.wmv', '.mkv',
    '.csv', '.xml', '.json', '.sql', '.db',
    '.xls', '.xlsx', '.ods',
    '.ppt', '.pptx', '.odp',
    '.html', '.htm', '.css', '.js',
    '.zip', '.rar', '.tar', '.gz',
    '.exe', '.dll', '.sys',
    '.py', '.java', '.c', '.cpp', '.h'
]

def profile(func: Callable) -> Callable:
    """A decorator that profiles a function's execution time.

    This decorator wraps a function to time its execution and records the time taken,
    along with the number of calls to the function, in a global dictionary.

    Args:
        func: The function to be profiled.

    Returns:
        A wrapped function with profiling functionality.

    """
    @functools.wraps(func)
    def wrapper_profile(*args: Any, **kwargs: Any) -> Any:
        """Wrapper function for the profile decorator.

        Args:
            *args: Variable length argument list for the decorated function.
            **kwargs: Arbitrary keyword arguments for the decorated function.

        Returns:
            The result of the decorated function.
        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        if func.__name__ not in profile_data:
            profile_data[func.__name__] = [elapsed_time, 1, None]
        else:
            profile_data[func.__name__][0] += elapsed_time
            profile_data[func.__name__][1] += 1
        return result
    return wrapper_profile


# @profile
def get_data(problem_files: bool = False) -> pd.DataFrame:
    """Retrieve data from local storage.

    Args:
        problem_files: A boolean flag used to determine which type of files to retrieve.

    Returns:
        A DataFrame containing the data of previously processed files or problematic files.
    """
    folder_name = "results"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    csv_file_path = 'results/gnazim_db_meta_data.csv'
    col_names = DATA_COL_NAMES  # Assuming data_col_names is defined elsewhere in your code
    if problem_files:
        csv_file_path = 'results/gnazim_db_problem_files.csv'
        col_names = PROBLEM_COL_NAMES  # Assuming problem_col_names is defined elsewhere in your code
    # Removed the duplicated condition
    if os.path.exists(csv_file_path):
        # df = pd.read_csv(csv_file_path,encoding='windows-1255') # in case pd.read without embedding doesnt work
        df = pd.read_csv(csv_file_path)
        df.fillna('', inplace=True)
    else:
        df = pd.DataFrame(columns=col_names)
    return df

# @profile
def establish_connection() -> GoogleDrive:
    """Establishes connection to Google Drive.

    Returns:
        An instance of GoogleDrive which is authenticated and ready to use.
    """
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)

# @profile
def find_four_digit_substring(string: str) -> str:
    """Finds a four-digit substring representing a year in a given string.

        Args:
            string: The string in which to search for a four-digit year.

        Returns:
            A string containing the four-digit year or an empty string if not found.
        """
    # current year, can be manually set or automatically taken from current time
    current_year = 2023

    # Pattern based on cases
    patterns = [
        r'(?<!\d)(1?\d{3})-(1?\d{3})(?!\d)',  # 1a
        r'(?<!\d)2[0][0-2][0-3]-(1?\d{3})(?!\d)|(?<!\d)(1?\d{3})-2[0][0-2][0-3](?!\d)',  # 1b
        r'(?<!\d)2[0][0-2][0-3]-2[0][0-2][0-3](?!\d)',  # 1c
        r'(?<!\d)(1?\d{3})(?!\d)',  # 2a
        r'(?<!\d)2[0][0-2][0-3](?!\d)'  # 2b
    ]

    for pattern in patterns:
        match = re.search(pattern, string)
        if match:
            year_group = match.group(0)
            # Filter out matches exceeding the current year
            years = [int(y) for y in year_group.split('-')]
            if all(year <= current_year for year in years):
                return year_group
    return ""

# @profile
def find_cd_label(string: str) -> str:
    """
    Finds a CD label in a given string.

    Args:
        string: The string in which to search for a CD label.

    Returns:
        A string containing the CD label or an empty string if not found.
    """
    pattern = r'(CD|D)?00\d+?\w?\s?'
    match = re.search(pattern, string)
    if match:
        return match.group()
    return ""

# @profile
def is_path_processed(processed_files_paths: set, path_to_check: str) -> bool:
    """
    Check if a given path is already present in the existing data DataFrame.

    Args:
        processed_files_paths: Set of paths of the already processed files.
        path_to_check: The path of the file to check.

    Returns:
        Boolean value indicating if the path is already processed.
    """
    # Check if path_to_check is in the set of paths
    return path_to_check in processed_files_paths

# @profile
def is_image(file_name: str) -> bool:
    """
    Check if the file name provided ends with a '.tif' extension indicating it is an image.

    Args:
        file_name: The name of the file to check.

    Returns:
        A boolean indicating whether the file is an image or not.
    """
    return file_name.endswith('.tif')

# @profile
def gcp_extract_text_from_image(data: dict[str, any], drive: GoogleDrive) -> dict[str, any]:
    """
    Extracts text from an image using GCP's OCR capabilities.

    Args:
        data: A dictionary containing at least the 'gcp_file_id'.
        drive: The GoogleDrive object used to access the file.

    Returns:
        A dictionary containing OCR metadata.
    """
    file_id = data["gcp_file_id"]
    # Download the file using PyDrive2
    file = drive.CreateFile({'id': file_id})
    file.GetContentFile("testname")
    # Load the downloaded file into a cv2 image in grayscale
    img = cv2.imread("testname", cv2.IMREAD_GRAYSCALE)
    processor = ImageProcessor(img)
    ocr_meta_data = processor.run()
    return ocr_meta_data

# @profile
def gcp_extract_years_author_type(preprocessed_dirpath: str) -> tuple[str, str, str]:
    """
    Extracts years, author, and type from a preprocessed directory path.

    Args:
        preprocessed_dirpath: A string of preprocessed directory path.

    Returns:
        A tuple containing years, author subject, and type subject.
    """
    years, author_subject, type_subject = "", "", ""
    for i, string in enumerate(reversed(preprocessed_dirpath)):
        preprocessed_str = string.replace(find_cd_label(string), '')
        years_candidate = find_four_digit_substring(preprocessed_str)
        if years == "" and years_candidate != "" and len(years_candidate) > 3:
            years = years_candidate

        preprocessed_str = preprocessed_str.replace(years, '')
        if len(preprocessed_str) > 1 and "," in preprocessed_str and type_subject == "" and i <= 1:
            if preprocessed_str.count("-") > 0:
                split_strings = [s.strip() for s in preprocessed_str.split("-", 1)]
                type_subject = next((s for s in split_strings if "," not in s), "")
                author_subject = next((s for s in split_strings if s != type_subject and "," in s), "")
            elif author_subject == "":
                author_subject = preprocessed_str.strip()
    return years, author_subject, type_subject

# @profile
def is_not_file(string: str) -> bool: # todo: check how fast this is
    """
    Check if the given string does not end with any of the file extensions listed in FILE_EXTENSIONS.

    Args:
        string (str): The string to check against the list of file extensions.

    Returns:
        bool: True if the string does not end with any of the listed file extensions, False otherwise.
    """
    return not any(string.endswith(ext) for ext in FILE_EXTENSIONS)

# @profile
def get_count(df: pd.DataFrame) -> int:
    """
    Get the count of rows in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame for which to count the rows.

    Returns:
        int: The number of rows in the DataFrame, or 0 if the DataFrame is empty.
    """
    if len(df) == 0:
        return 0
    else:
        return df.shape[0]

# @profile
def gcp_process_files_in_folder(folder_id: str, drive: GoogleDrive, scanned_files_amount_now: int, processed_files_paths: list[str], folder_title: str = "") -> list[tuple[str, str]]:
    """
    Process all files in a given folder on Google Drive, updating processed files and handling reconnections.

    Args:
        folder_id (str): The ID of the folder whose files are to be processed.
        drive (GoogleDrive): The GoogleDrive object used to interact with Google Drive.
        scanned_files_amount_now (int): The number of files already scanned.
        processed_files_paths (List[str]): A list of paths to files that have already been processed.
        folder_title (str, optional): The title of the folder. Defaults to an empty string.

    Returns:
        List[Tuple[str, str]]: A list of tuples where each tuple contains the ID and title of a folder that needs to be processed.
    """

    folder_files_list = drive.ListFile({'q': "'%s' in parents and trashed=false" % folder_id}).GetList()
    problem_cnt = get_count(get_data(problem_files=True))
    sample_cnt = scanned_files_amount_now
    current_data = get_data()
    # List to hold folder parameters for non-image files
    folders_to_process = []
    new_processed_files = pd.DataFrame(columns=DATA_COL_NAMES)
    start_time = time.time()
    print(f"Start Attempt to Process Folder: {folder_title}")
    i = 0
    for file in folder_files_list:
        i += 1
        try:
            current_folder_title = folder_title + "\\" + file['title']
            new_row = []
            if is_image(file['title']):
                if not is_path_processed(processed_files_paths, current_folder_title):
                    new_row = gcp_process_file(current_folder_title, drive, file, sample_cnt, folder_title)
                    sample_cnt += 1
                    new_processed_files = pd.concat([new_processed_files, new_row], axis=0)
                    print(f"Process file:{file['title']} First attempt Successfully")
            elif is_not_file(file['title']):
                folders_to_process.append((file['id'], current_folder_title))

        except Exception as e:
            if str(e) == "invalid_grant: Bad Request":
                try:
                    print(f"Process file:{file} First attempt failed! Connection Problem, Starting Second Attempt...")
                    gcp_reconnect(drive)
                    new_row = gcp_process_file(current_folder_title, drive, file, sample_cnt, folder_title)
                    sample_cnt += 1
                    new_processed_files = pd.concat([new_processed_files, new_row], axis=0)
                    continue
                except:
                    print(f"ReProcess file:{file} Second attempt failed! Add To Problems...\n")
            new_row = create_new_problem_row(current_folder_title, e, file, new_row, problem_cnt)
            write_problem_folder_to_csv(new_row)
            raise e
    end_time = time.time()

    if sample_cnt > scanned_files_amount_now:
        # new_processed_files.to_csv('results/gnazim_db_meta_data.csv', mode='a', header=False, index=False,encoding='windows-1255')
        new_processed_files.to_csv('results/gnazim_db_meta_data.csv', mode='a', header=False, index=False,encoding='windows-1255')

        print(f"Folder {folder_title} Processed Successfully and Save to Disk after {end_time - start_time} time\n")
    else:
        print(
            f"Folder {folder_title} Processed Successfully "
            f"and No Files Saved to Disk after {end_time - start_time} seconds\n")
    return folders_to_process

# @profile
def gcp_reconnect(drive: GoogleDrive) -> None:
    """
    Attempt to reconnect to Google Drive service.

    Args:
        drive (GoogleDrive): The GoogleDrive object to refresh the connection for.

    Returns:
        None
    """
    drive.Refresh()
    print(f"Connection attempt failed. Retrying in {60} seconds...")
    time.sleep(60)

# @profile
def create_new_problem_row(current_folder_title: str, e: Exception, file: dict[str, any], new_row: dict[str, any], problem_cnt: int) -> dict[str, any]:
    """
    Create a new row for logging problematic files when exceptions occur during processing.

    Args:
        current_folder_title (str): The title of the current folder being processed.
        e (Exception): The exception that was raised during processing.
        file (Dict[str, Any]): The file that caused the exception.
        new_row (Dict[str, Any]): The initial new row dictionary where the problem data will be stored.
        problem_cnt (int): The current count of problematic files.

    Returns:
        Dict[str, Any]: A dictionary representing the new row with the problem file's data.
    """
    try:
        problem_cnt += 1
        # Handle exceptions by appending the problematic folder and file to 'gnazim_db_problem_folders.csv'
        print(f"Exception number {problem_cnt}!\nencountered for folder: {current_folder_title}\n"
              f"file: {file['title']}\nError: {e}\n")
        new_row.update({"folder_name": [current_folder_title],
                        "file_name": [file['title']],
                        "error_message": str(e)})
    except Exception as error:
        new_row = {key: "" for key in DATA_COL_NAMES}
        new_row.update({"folder_name": [current_folder_title],
                        "file_name": [file['title']],
                        "error_message": str(e)})
    return new_row

# @profile
def gcp_process_file(file_path: str, drive: any, file1: dict[str, any], count: int, folder_title: str) -> pd.DataFrame:
    """
    Process a single file from Google Cloud Platform, extracting meta and OCR data.

    Args:
        file_path (str): The path to the file being processed.
        drive (Any): The GoogleDrive object used to interact with Google Drive.
        file1 (Dict[str, Any]): The file object to process.
        count (int): The current count of processed files.
        folder_title (str): The title of the folder containing the file.

    Returns:
        pd.DataFrame: A DataFrame with one row of processed data for the file.
    """
    # initialize new row
    new_row = {key: "" for key in DATA_COL_NAMES}

    # Process Meta Data From Folder Structure
    new_row["path"] = file_path
    new_row["gcp_file_id"] = file1['id']
    new_row["folder_name"] = folder_title
    new_row["gcp_folder_id"] = file1['parents'][0]['id']
    new_row["file_name"] = file1['title']
    processed_path = new_row["path"].split("\\")
    processed_path.pop(0)
    new_row["years"], new_row["author_subject"], new_row["type"] = gcp_extract_years_author_type(processed_path)
    count += 1
    new_row["Years"] = ""  # this is redundant column we need to remove so in the end of the run will do
    new_row["identifier"] = f"IDGNAZIM000{count}"

    # Process OCR Data From Image
    ocr_meta_data = gcp_extract_text_from_image(new_row, drive)
    # print("-" * 110)
    # print(f"-----------Meta data for file number {count} is:"
    #       f"----------------------------------------------------------------")
    # for k in ocr_meta_data.keys():
    #     print(k, ": ", ocr_meta_data[k])
    # print("-" * 110)
    # print()

    # Save All Processed New Row Data
    new_row.update(ocr_meta_data)  # Update new_data dictionary with ocr_meta_data
    new_row = pd.DataFrame(new_row, index=[0])  # specify the index explicitly

    return new_row

# @profile
def run() -> None:
    """
    Run the main processing function to iterate over folders and process files from Google Cloud Platform.

    Returns:
        None: This function does not return any value.
    """
    run_start_time = time.time()

    # GCP Connection
    current_drive = establish_connection()
    # Initialize Folders root
    folders_to_process = get_folders_queue()
    reconnect_trashold = 600
    scanned_files_trashold = 2000
    problem_folders_to_process = []  # Store problematic folders' data here
    data = get_data()
    scanned_files_amount_in_beginning = get_count(data)
    gcp_processed_folders_names = set(data['folder_name'].unique())

    # Create a set of all values of "path" to later check no file that have been reprocessed
    # to avoid duplicates.
    processed_files_paths = set(data["path"].values)

    while folders_to_process:

        current_folder_id, current_folder_title = folders_to_process.pop(0)

        data = get_data()
        scanned_files_amount_now = get_count(data)
        print(
            f"\nscanned_files_amount_in_beginning:{scanned_files_amount_in_beginning} , scanned_files_amount_now: {scanned_files_amount_now}")

        scanned_files_amount = scanned_files_amount_now - scanned_files_amount_in_beginning
        if scanned_files_amount >= scanned_files_trashold:
            # Save the Queue To Tackle the longer BFS Problem to able to start where we stoped
            folders_to_process_queue = pd.DataFrame(folders_to_process, columns=['folder_id', 'files'])
            folders_to_process_queue.to_csv('results/gnazim_folders_to_process_queue.csv',
                                            mode='w', header=True,index=False,encoding='windows-1255')
            break

        if scanned_files_amount >= reconnect_trashold:  # Reconnect Every 600 read samples
            reconnect_trashold += reconnect_trashold
            gcp_reconnect(current_drive)

        try:
            new_folders_to_process = gcp_process_files_in_folder(current_folder_id, current_drive,
                                                                 scanned_files_amount_now, processed_files_paths,
                                                                 current_folder_title)
            new_filtered_folders_to_process = [(folder_id, folder_title) for (folder_id, folder_title) in
                                               new_folders_to_process if
                                               folder_title not in gcp_processed_folders_names]
            folders_to_process.extend(new_filtered_folders_to_process)

        except Exception as er:
            if str(er) == "invalid_grant: Bad Request":
                print(f"Error processing folder {current_folder_id} - {current_folder_title}: {er}")
                print("***** Finished Run due to GCP Connection Problem *****")
                break
            else:
                # Log and save other exceptions for later inspection
                print(f"Error processing folder {current_folder_id} - {current_folder_title}: {er}")
                print("***** Continue Run to Different Folder, Problem Been Saved For Future Fix *****")
                # problem_folders_to_process.append(
                #     {"folder_id": current_folder_id, "folder_title": current_folder_title})
                # write_problem_folder_to_excel(problem_folders_to_process)
    #
    run_end_time = time.time()
    print(f"End Of Run:\nProblem folders:\n{problem_folders_to_process}\n"
          f"Run function finished processing {scanned_files_amount} files "
          f"in {(run_end_time - run_start_time) / 60} minutes!")


def get_folders_queue():
    root_folder_id = "1sciJWxtjAxbas-fyxuIiXPANOXGxectN"  # First Folder to Preprocess Than DFS OR BFS On Tree
    csv_folders_queue_path = 'results/gnazim_folders_to_process_queue.csv'
    folders_queue_col_names = ["folder_id", "files"]
    df = pd.DataFrame(columns=folders_queue_col_names)
    if os.path.exists(csv_folders_queue_path):
        df = pd.read_csv(csv_folders_queue_path)
        df.fillna('', inplace=True)
    if not df.empty:
        folders_to_process = [(row['folder_id'], row['files']) for index, row in df.iterrows()]
    else:
        folders_to_process = [(root_folder_id, "")]
    return folders_to_process


# @profile
def write_problem_folder_to_excel(new_problem_row: dict[str, any]) -> None:
    """
    Write problem folder information to an Excel file.

    Args:
        new_problem_row (Dict[str, Any]): The data of the new problem row to be written to Excel.

    Returns:
        None: This function does not return any value.
    """
    # If there are problematic folders, save them to Excel
    if new_problem_row:
        new_problem_data = pd.DataFrame(new_problem_row)

        problem_folders = get_data(problem_files=True)
        if problem_folders.shape[0] == 0:
            new_problem_folders = new_problem_data
        else:
            new_problem_folders = pd.concat([problem_folders, new_problem_data], axis=0, ignore_index=True)

        with pd.ExcelWriter('results/gnazim_db_problem_files.xlsx') as writer:
            new_problem_folders.to_excel(writer, index=False, sheet_name='Sheet1')

# @profile
def write_problem_folder_to_csv(new_problem_row: dict[str, any]) -> None:
    """
    Write problem folder information to a CSV file.

    Args:
        new_problem_row (Dict[str, Any]): The data of the new problem row to be written to CSV.

    Returns:
        None: This function does not return any value.
    """
    # If there are problematic folders, save them to CSV
    if new_problem_row:
        new_problem_data = pd.DataFrame(new_problem_row)
        csv_file_path = 'results/gnazim_db_problem_files.csv'

        # Check if the CSV file already exists to decide whether to write headers
        write_header = not os.path.exists(csv_file_path)

        # Append the new data to the CSV file (or create a new file if it doesn't exist)
        # new_problem_data.to_csv(csv_file_path, mode='a', index=False, header=write_header,encoding='windows-1255')
        new_problem_data.to_csv(csv_file_path, mode='a', index=False, header=write_header,encoding='windows-1255')

def create_df_gcp_file_links():
    """
      Reads a CSV file containing metadata from a specific file path and adds two new columns with generated Google Cloud
      Platform (GCP) file and folder links. The new data is saved to a new CSV file.

      The function internally defines two helper functions:
      - `create_gcp_file_link`: Generates a link to the file in Google Drive based on the 'gcp_file_id' in each row.
      - `create_gcp_folder_link`: Generates a link to the folder in Google Drive based on the 'gcp_folder_id' in each row.

      These links are then applied to each row of the DataFrame, creating two new columns 'gcp_image_link' and
      'gcp_folder_link'. Finally, the updated DataFrame is saved as a new CSV file with the added link columns.

      Raises:
          FileNotFoundError: If the input CSV file does not exist at the specified `file_path`.
          Exception: For issues related to reading the CSV, applying functions, or writing the output file.

      CSV File structure:
          The input CSV is expected to have at least 'gcp_file_id' and 'gcp_folder_id' columns.

      Side effects:
          - Reads a CSV file from a hardcoded file path.
          - Creates and saves a new CSV file with additional columns to a hardcoded file path.
          - The function does not return any value or the modified DataFrame.
      """
    data=get_data()

    def create_gcp_file_link(row):
        link = f"https://drive.google.com/file/d/{row['gcp_file_id']}"
        return link

    def create_gcp_folder_link(row):
        link = f"https://drive.google.com/drive/folders/{row['gcp_folder_id']}"
        return link

    # Apply the function to each row and create a new column 'gcp_image_link' with the resulting links
    data['gcp_image_link'] = data.apply(create_gcp_file_link, axis=1)
    data['gcp_folder_link'] = data.apply(create_gcp_folder_link, axis=1)
    file_path = r'C:\Users\yarin\PycharmProjects\DHC\GnazimProject\results\gnazim_db_meta_data_with_links.csv'
    # Save the DataFrame to a CSV file
    data.to_csv(file_path, index=False, encoding='windows-1255')  # Set index=False if you don't want to save the indexa=3



# ************************************************************************************************************************

if __name__ == "__main__":
    # Run start timestamp
    run_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Run the main function
    run()

    # Run end timestamp
    run_end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    create_df_gcp_file_links()
    # Add the run timestamp to the profile data
    for func_data in profile_data.values():
        func_data[2] = run_end_time  # Update the run_timestamp for each function

    # # Convert the profile data to a DataFrame
    # profile_df = pd.DataFrame.from_dict(profile_data, orient='index', columns=['Total Time', 'Calls', 'Run Timestamp'])
    # profile_df.sort_values('Total Time', ascending=False, inplace=True)
    #
    # # Save to a CSV for later analysis
    # profile_df.to_csv('results/gnazim_function_profile_data.csv')
    #
    # # Display the DataFrame
    # print(profile_df)
