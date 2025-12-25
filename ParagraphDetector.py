# Based on
# https://stackoverflow.com/questions/57249273/how-to-detect-paragraphs-in-a-text-document-image-for-a-non-consistent-text-stru


import cv2
import re
import pytesseract


class ParagraphDetector:
    """
    A class for processing images, particularly for OCR (Optical Character Recognition)
    and detection of text regions.

    This class provides a set of methods to preprocess images, detect regions of interest (ROIs),
    extract text using OCR, and determine characteristics like whether the text is handwritten.
    It's designed to work primarily with Hebrew text but can be adapted for other languages.

    Attributes:
        image: The image to be processed.
        lang (str): The language code used for OCR processing, defaulting to Hebrew ('heb').
        ocr_of_all_text_no_prero (str): Stores the OCR result of the image without preprocessing.
        ocr_of_all_text_prepro (str): Stores the OCR result of the image after preprocessing.
        rectangles_detected_image: Stores the image with detected rectangles marked.

    Methods:
        __init__(self, image, lang='heb'): Initializes the ImageProcessor instance.
        minimum_abs_difference(rect1, rect2): Static method to calculate the minimum absolute
            difference between two rectangles.
        exists_two_close_rectangles(rectangles): Checks if any two rectangles in a list are close to each other.
        get_paragraph_boundingbox(): Extracts paragraph bounding boxes from the image.
        resize_and_plot(image, show_image=False): Static method to resize and optionally display an image.
        create_croped_roi(contours, image): Creates cropped regions of interest based on contours.
        post_contour_preprocess(dilate): Performs post-contour detection preprocessing on the image.
        roi_detection(thresh): Detects regions of interest in the image.
        pre_contour_preprocess(): Prepares the image for contour detection.
        plot_roi_and_ocr(cropped_images): Processes cropped images to extract OCR data and plot ROIs.
        is_handwritten(): Determines if the text is handwritten based on the proportion of Hebrew characters.
        run(log_metadata=False): Executes the main processing pipeline of the ImageProcessor class.
    """
    def __init__(self, image, lang='heb'):
        """
        Initializes the ImageProcessor class with the provided image and language.

        This method sets the initial state of the ImageProcessor instance by storing the given image
        and language. It also calls the 'resize_and_plot' method to resize and optionally display the image.

        Args:
            image: An image object to be processed.
            lang (str, optional): The language code for OCR processing, defaulting to Hebrew ('heb').
        """
        self.image = image
        self.lang = lang
        self.resize_and_plot(self.image, show_image=False)

    @staticmethod
    def minimum_abs_difference(rect1, rect2):
        """
         Calculates the minimum absolute difference between the edges of two rectangles.

         This static method computes the smallest distance between the edges of two given rectangles. It's
         used to determine the proximity of rectangles in image processing.

         Args:
             rect1: A tuple representing the first rectangle (x, y, width, height).
             rect2: A tuple representing the second rectangle (x, y, width, height).

         Returns:
             int: The minimum absolute difference between the edges of the two rectangles.
         """
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        differences = [abs(x1 - x2), abs((x1 + w1) - x2), abs(x1 - (x2 + w2)), abs((x1 + w1) - (x2 + w2))]
        return min(differences)

    def exists_two_close_rectangles(self, rectangles):
        """
           Checks if any two rectangles in a list are close to each other based on a threshold.

           This method iterates through a list of rectangles and uses 'minimum_abs_difference' to check if
           any two rectangles are closer than a certain threshold, indicating potential overlap or closeness.

           Args:
               rectangles: A list of rectangles (each a tuple of x, y, width, height).

           Returns:
               bool: True if any two rectangles are close to each other, False otherwise.
           """
        height, width = self.image.shape
        height_epsilon, width_epsilon = height * 0.002, width * 0.002
        for i, rect1 in enumerate(rectangles):
            for rect2 in rectangles[i + 1:]:
                min_distance = self.minimum_abs_difference(rect1, rect2)
                if min_distance < height_epsilon or min_distance < width_epsilon:
                    return True
        return False

    def get_paragraph_bounding_box(self):
        """
          Extracts paragraph bounding boxes from the image using OCR and contour detection.

          This method performs several image preprocessing steps to detect and extract paragraphs from the
          image. It applies OCR both before and after preprocessing to extract text, and identifies the
          bounding boxes of paragraphs in the image.

          Returns:
              list: A list of cropped images representing the detected paragraphs, along with their
                    coordinates and OCR extracted text.
          """
        self.ocr_of_all_text_no_prero = re.sub(r'\n+', ' ', pytesseract.image_to_string(self.image, lang=self.lang))
        # print(f"\nocr of all text with NO preprocess: {self.ocr_of_all_text_no_prero}\n")

        thresh = self.pre_contour_preprocess()
        contours, dilate, rectangles = self.roi_detection(thresh)
        thresh = self.post_contour_preprocess(dilate)
        self.ocr_of_all_text_prepro = re.sub(r'\n+', ' ', pytesseract.image_to_string(thresh, lang=self.lang))
        # print(f"ocr of all text with preprocess: {self.ocr_of_all_text_prepro}\n")
        return self.create_cropped_roi(contours, thresh)

    @staticmethod
    def resize_and_plot(image, show_image=False):
        """
        Resizes and optionally displays an image.

        This static method resizes the given image to fit a predefined screen resolution while maintaining
        aspect ratio. It can optionally display the resized image in a window.

        Args:
            image: The image to be resized and displayed.
            show_image (bool, optional): Flag to indicate whether to display the image, defaults to False.
        """
        screen_res = 1920, 1080
        scale_width = screen_res[0] / image.shape[1]
        scale_height = screen_res[1] / image.shape[0]
        scale = min(scale_width, scale_height) * 0.5
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        dim = (width, height)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        if show_image:
            cv2.imshow('Resized Image', resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # if name:
        #     # Construct the full path where the image will be saved
        #     file_path = rf"C:\Users\yarin\PycharmProjects\DHC\GnazimProject\outputs\{name}.png"
        #
        #     # Save the image
        #     cv2.imwrite(file_path, resized)

    def create_cropped_roi(self, contours, image):
        """
         Creates and returns cropped regions of interest (ROIs) based on contours.

         This method processes the given contours to identify regions of interest in the provided image.
         It crops these regions and extracts OCR text from them, returning the cropped images, their
         coordinates, and the extracted text.

         Args:
             contours: Contours detected in the image.
             image: The image from which the ROIs are to be cropped.

         Returns:
             list: A list of tuples containing the cropped images, their coordinates, and OCR text.
         """
        copy_image = image.copy()
        cropped_images = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 15 and h > 40:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cropped = copy_image[y:y + h, x:x + w]
                cropped_images.append(
                    (cropped, (x, y, x + w, y + h), pytesseract.image_to_string(cropped, lang=self.lang)))
                # cropedimage, top left and bottom right corner of image in original images,ocr text
        self.resize_and_plot(image, show_image=False)
        self.rectangles_detected_image = image
        return cropped_images[::-1]

    def post_contour_preprocess(self, dilate):
        """
          Performs post-contour detection preprocessing on the image.

          This method applies various image processing techniques, such as median blurring and thresholding,
          to the dilated image obtained after contour detection. This helps in enhancing the features of
          the image for better OCR results.

          Args:
              dilate: The dilated image obtained after initial contour detection.

          Returns:
              ndarray: The processed image after applying post-contour detection techniques.
          """
        test = (-self.image + dilate)
        test = cv2.medianBlur(test, 5)
        test = cv2.medianBlur(test, 5)
        test = cv2.threshold(test, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return test

    def roi_detection(self, thresh):
        """
          Detects regions of interest (ROIs) in the image based on thresholding and contour detection.

          This method identifies contours in the thresholded image and iteratively dilates the image to
          merge close contours. It returns the final contours, the dilated image, and the bounding rectangles
          of the detected ROIs.

          Args:
              thresh: The thresholded image for contour detection.

          Returns:
              tuple: A tuple containing the contours, the final dilated image, and the rectangles of ROIs.
          """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        iter_param = 4
        dilate = cv2.dilate(thresh, kernel, iterations=iter_param)
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = [cv2.boundingRect(c) for c in contours]
        flag = self.exists_two_close_rectangles(rectangles)
        self.resize_and_plot(dilate, show_image=False)

        while flag:
            # self.resize_and_plot(dilate)
            iter_param += 2
            dilate = cv2.dilate(thresh, kernel, iterations=iter_param)
            contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rectangles = [cv2.boundingRect(c) for c in contours]
            flag = self.exists_two_close_rectangles(rectangles)
        self.resize_and_plot(dilate, show_image=False)

        return contours, dilate, rectangles

    def pre_contour_preprocess(self):
        """
         Performs preprocessing on the image before contour detection.

         This method applies a series of median and Gaussian blurs to the image, followed by thresholding,
         to prepare it for contour detection. This preprocessing is crucial for effective contour detection.

         Returns:
             ndarray: The preprocessed image ready for contour detection.
         """
        blur = cv2.medianBlur(self.image, 5)
        blur = cv2.GaussianBlur(blur, (5, 5), 0)
        thresh = cv2.medianBlur(blur, 5)
        thresh = cv2.medianBlur(thresh, 5)
        thresh = cv2.medianBlur(thresh, 5)
        thresh = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return thresh

    def plot_roi_and_ocr(self, cropped_images):
        """
          Processes cropped images to extract OCR data and plot ROIs.

          This method takes the cropped images obtained from contour detection, extracts OCR data, and plots
          the regions of interest. It organizes the OCR data based on its position and content.

          Args:
              cropped_images: A list of cropped images along with their coordinates and OCR text.

          Returns:
              dict: A dictionary containing the extracted OCR data and metadata.
          """
        # Initializing the dictionary with default values
        paragraphs_detection_process_completed = 1
        extracted_ocr_data = {
            "ocr_written_on": "",
            "ocr_written_by": "",
            "ocr_main_content": "",
            "ocr_additional_content": "",
            "ocr_written_on_coords": "",
            "ocr_written_by_coords": "",  # Add coordinates for 'written_by'
            "ocr_main_content_coords": "",  # Add coordinates for 'main_content'
            "ocr_additional_content_coords": "",  # Add coordinates for 'additional_content'
            "paragraphs_detection_successes": paragraphs_detection_process_completed}
        image_width = self.image.shape[1]  # Image width is the second value of shape
        middle_x_point = image_width / 2

        # Create ocr_list by filtering out items where ocr text was found in the bounding box(len(item[2]) < 1 )
        ocr_list = list(filter(lambda item: len(item[2]) > 0, cropped_images))

        # If written_on_candidates list is not empty, process it
        if ocr_list:
            # Create written_on_candidates list
            if len(ocr_list) > 1:
                written_on_candidates = sorted(
                    filter(lambda item: item[1][0] > middle_x_point, ocr_list),
                    key=lambda item: (item[1][1], -item[1][0])
                )
            else:
                written_on_candidates = ocr_list
            if written_on_candidates:
                written_on_image, written_on_coords, written_on_text = written_on_candidates[0]
                written_on_text = [line for line in written_on_text.split("\n") if line.strip()]
                extracted_ocr_data["ocr_written_on"] = written_on_text[0]
                extracted_ocr_data["ocr_written_on_coords"] = re.sub(r'[()]', '', str(written_on_coords))
                self.resize_and_plot(written_on_image, show_image=False)
                if len(written_on_text) > 1:
                    extracted_ocr_data["ocr_written_by"] = written_on_text[1]
                    # Save the coordinates for 'written_by' as well.
                    extracted_ocr_data["ocr_written_by_coords"] = re.sub(r'[()]', '', str(written_on_coords))
            else:
                paragraphs_detection_process_completed = 0
            # Create main_content_candidates list
            main_content_candidates = sorted(ocr_list, key=lambda x: len(x[2]), reverse=True)
            if main_content_candidates:
                main_content_image, main_content_coords, main_content_text = main_content_candidates[0]
                extracted_ocr_data["ocr_main_content"] = main_content_text.replace("\n", "")
                # Save the coordinates for 'main_content'.
                extracted_ocr_data["ocr_main_content_coords"] = re.sub(r'[()]', '', str(main_content_coords))
                self.resize_and_plot(main_content_image, show_image=False)
                # Check if the next longest text is different from 'written_on' text
                if len(main_content_candidates) > 1:
                    additional_content_image, additional_content_coords, additional_content_text = \
                        main_content_candidates[1]
                    if additional_content_text != extracted_ocr_data["ocr_written_on"] + extracted_ocr_data[
                        "ocr_written_by"]:
                        extracted_ocr_data["ocr_additional_content"] = additional_content_text
                        # Save the coordinates for 'additional_content'.
                        extracted_ocr_data["ocr_additional_content_coords"] = re.sub(r'[()]', '',
                                                                                     str(additional_content_coords))
                        self.resize_and_plot(additional_content_image, show_image=False)
            else:
                paragraphs_detection_process_completed = 0
        else:
            self.resize_and_plot(self.rectangles_detected_image, show_image=False)  # SHOW IMAGE THAT DETECTION FAILED
            paragraphs_detection_process_completed = 0
        extracted_ocr_data["paragraphs_detection_successes"] = paragraphs_detection_process_completed
        return extracted_ocr_data

    def is_handwritten(self):
        """
        Determines if the text is handwritten based on the proportion of Hebrew characters.

        This function first selects the appropriate text to analyze: it uses `self.ocr_of_all_text_prepro`
        if it's not empty; otherwise, it falls back to `self.ocr_of_all_text_no_prero`. It then calculates
        the proportion of Hebrew characters in the selected text. The text is considered to be handwritten
        if the proportion of Hebrew characters is less than a specified threshold (currently set at 0.4).

        The function removes all non-Hebrew characters from the selected text and then compares the length
        of the filtered text to the length of the original text. If the length of the filtered text is less
        than 40% of the original text's length, it returns 1 (indicating handwritten). Otherwise, it returns 0.

        Example:
            If self.ocr_of_all_text_prepro is empty and self.ocr_of_all_text_no_prero = 'סי | כב // / ָ 9 ה [ערה, ס.ל, / ',
            then filtered_hebrew_text = 'סיכבָהערהסל'.
            The length of filtered_hebrew_text is 11 and the length of the original text is 32.
            Since 11 < 32 * 0.4 (which equals 12.8), the function returns 1.

        Returns:
            int: 1 if the text is considered handwritten (based on the proportion of Hebrew characters),
                 0 otherwise.
        """
        proportion_value = 0.4
        # Remove all non-Hebrew characters
        if len( self.ocr_of_all_text_prepro) == 0:
            hebrew_text = self.ocr_of_all_text_no_prero
        else:
            hebrew_text = self.ocr_of_all_text_prepro

        filtered_hebrew_text = re.sub(r'[^\u0590-\u05FF]', '', hebrew_text)
        return 1 if len(filtered_hebrew_text) < len(hebrew_text) * proportion_value else 0

    def run(self, log_metadata=False):
        """
         Executes the main processing pipeline of the ImageProcessor class.

         This method orchestrates the entire processing flow, including paragraph detection, ROI plotting,
         OCR extraction, and handwritten text detection. It compiles the results and, if requested, logs
         detailed metadata.

         Args:
             log_metadata (bool, optional): Flag to indicate whether to log metadata, defaults to False.

         Returns:
             dict: A dictionary containing all extracted data and results from the processing pipeline.
         """

        roi_list = self.get_paragraph_bounding_box()
        result = self.plot_roi_and_ocr(roi_list)
        result["ocr_all_text_preprocess"] = self.ocr_of_all_text_prepro
        result['is_handwritten'] = self.is_handwritten()
        result["ocr_all_text_no_preprocess"] = self.ocr_of_all_text_no_prero
        if result['paragraphs_detection_successes'] and len(result["ocr_written_by"]) + len(result["ocr_written_on"]) > 1:
            cut_index = len(result["ocr_written_by"]) + len(result["ocr_written_on"]) + len(
                result["ocr_additional_content"])
            result["ocr_main_content_all_text_preprocess"] = self.ocr_of_all_text_prepro[cut_index:]
            result["ocr_main_content_all_text_no_preprocess"] = self.ocr_of_all_text_no_prero[cut_index:]
        else:
            result["ocr_main_content_all_text_preprocess"] = ""
            result["ocr_main_content_all_text_no_preprocess"] = ""

        if log_metadata:
            print("-" * 110)
            print(
                "--------OCR extracted meta data from image is:----------------------------------------------------------------")
            for k in result.keys():
                print(k, ": ", result[k])
            print("-" * 110)
            print()
        return result


# Unittest I made for ImageProcessor class on different samples :
# Samples:
# path = 'POC_sample3_handwriten.tif'  # Hand written Example ALSO WORKS - GOOD
# path = 'POC_sample2_withnoise.tif'  # Typed with noise - GOOD
# path = 'SAMPLE_Long_Author04.tif'# Typed with long author name - GOOD
# path = 'POC_sample5_withnoise_Author.tif'  #  FIXED PROBLEM OVERLAP RECTANGLES - GOOD

# The code of the test:
# img=cv2.imread(path, 0)
# processor = ImageProcessor(img)
# r=processor.run(True)
# print(r)

