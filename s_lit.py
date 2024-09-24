import streamlit as st
from PIL import Image
from paddleocr import PaddleOCR
import numpy as np
import requests

def download_image(image_url, save_path="downloaded.jpg"):
    # Send a GET request to the image URL
    response = requests.get(image_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open a file in write-binary mode and save the image content
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Image successfully downloaded and saved to {save_path}")
    else:
        print(f"Failed to retrieve the image. Status code: {response.status_code}")





# Initialize the PaddleOCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # You can specify 'ch', 'en', 'french', etc.

def extract_text(image_path):
    # Extract the OCR result from the image
    result = ocr.ocr(image_path)

    # Process and display the results
    extracted_text = []
    for line in result:
      if line:
        for text_info in line:
            text = text_info[1][0]  # Extract the recognized text
            extracted_text.append(text)
    # print(extracted_text)
    return extracted_text


def text_centre(bounding_box):
    # Bounding box coordinates: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    # bounding_box = [[59.0, 259.0], [104.0, 255.0], [107.0, 283.0], [61.0, 287.0]]

    # Convert the list to a numpy array for easy manipulation
    bounding_box = np.array(bounding_box)

    # Calculate the center of the bounding box
    center_x = np.mean(bounding_box[:, 0])  # Average of x-coordinates
    center_y = np.mean(bounding_box[:, 1])  # Average of y-coordinates

    # Print the center
    center = (center_x, center_y)
    return center


def extract_wit_position():
  img_path = 'downloaded.jpg'

# Perform OCR on the image
  result = ocr.ocr(img_path, cls=True)
  ret=[]
  # Iterate through the results and print text positions and the recognized text
  if result[0]:
    for line in result[0]:
        # Each line contains a bounding box and the text
        bbox, text_info = line
        c=text_centre(bbox)
        ret.append({"text":text_info[0],"center":c})
        # print(f"Bounding Box: {bbox}")
        # print(f"Text: {text_info[0]}")

  return ret

# extract_wit_position()



entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'},
    'maximum_weight_recommendation': {'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre',
        'cubic foot',
        'cubic inch',
        'cup',
        'decilitre',
        'fluid ounce',
        'gallon',
        'imperial gallon',
        'litre',
        'microlitre',
        'millilitre',
        'pint',
        'quart'}
}

allowed_units = {unit for entity in entity_unit_map for unit in entity_unit_map[entity]}
allowed_units=list(allowed_units)

orgasm={
    "width": {"centimetre", "cm", "centimeter", "foot","feet", "ft", "inch", "in", "metre","meter", "m", "millimetre","millimeter", "mm", "yard", "yd"},
    "depth": {"centimetre", "cm", "centimeter", "foot","feet", "ft", "inch", "in", "metre","meter", "m", "millimetre","millimeter", "mm", "yard", "yd"},
    "height": {"centimetre", "cm", "centimeter", "foot","feet", "ft", "inch", "in", "metre","meter", "m", "millimetre","millimeter", "mm", "yard", "yd"},
    "item_weight": {"gram", "g", "grams","gsm","gm","gms" "kilogram", "kg", "kilograms","kgs", "microgram", "µg", "micrograms", "milligram", "mg", "milligrams", "ounce", "oz", "ounces", "pound", "lb","lbs", "pounds", "ton", "tons"},
    "maximum_weight_recommendation": {"gram", "g", "grams","gsm","gm","gms" "kilogram", "kg", "kilograms","kgs", "microgram", "µg", "micrograms", "milligram", "mg", "milligrams", "ounce", "oz", "ounces", "pound", "lb","lbs", "pounds", "ton", "tons"},
    "voltage": {"kilovolt", "kv", "kilo volt", "kilovolts","kilo volts", "millivolt", "mv", "milli volt", "millivolts","milli volts", "volt", "v", "volts"},
    "wattage": {"kilowatt", "kw", "kilo watt", "kilowatts","kilo watts", "watt", "w", "watts"},
    "item_volume": {"centilitre", "cl", "centiliter", "cubic foot", "cu ft", "cubic inch", "cu in", "cup", "cups", "decilitre", "dl", "deciliter", "fluid ounce", "fl oz", "fluid ounces", "gallon", "gallons", "imperial gallon", "imperial gallons", "litre", "l", "liter", "microlitre", "µl", "microliter", "millilitre", "ml", "milliliter", "pint", "pints", "quart", "quarts"}
}


mapp={'centilitre': 'centilitre', 'pints': 'pint',"'":'inch', "''": 'inch', '"': 'inch', 'cubic inch': 'cubic inch', 'cup': 'cup', 'fl oz': 'fluid ounce', 'milliliter': 'millilitre', 'imperial gallon': 'imperial gallon', 'cl': 'centilitre', 'liter': 'litre', 'fluid ounces': 'fluid ounce', 'centiliter': 'centilitre', 'ml': 'millilitre', 'quarts': 'quart', 'cubic foot': 'cubic foot', 'gallon': 'gallon', 'imperial gallons': 'imperial gallon', 'cu in': 'cubic inch', 'decilitre': 'decilitre', 'deciliter': 'decilitre', 'l': 'litre', 'millilitre': 'millilitre', 'litre': 'litre', 'cups': 'cup', 'cu ft': 'cubic foot', 'gallons': 'gallon', 'microlitre': 'microlitre', 'microliter': 'microlitre', 'dl': 'decilitre', 'µl': 'microgram', 'quart': 'quart', 'fluid ounce': 'fluid ounce', 'fl': 'fluid ounce', 'pint': 'pint', 'w': 'watt', 'ws': 'watt', 'watt': 'watt', 'watts': 'watt', 'kilowatts': 'kilowatt', 'kilowatt': 'kilowatt', 'kilo watt': 'kilowatt', 'kw': 'kilowatt', 'kilo watts': 'kilowatt', 'kilovolts': 'kilovolt', 'volt': 'volt', 'kilo volts': 'kilovolt', 'millivolt': 'millivolt', 'v': 'volt', 'millivolts': 'millivolt', 'milli volt': 'millivolt', 'kilovolt': 'kilovolt', 'volts': 'volt', 'kv': 'kilovolt', 'milli volts': 'millivolt', 'kilo volt': 'kilovolt', 'mv': 'millivolt', 'ounce': 'ounce', 'milligrams': 'milligram', 'pounds': 'pound', 'kgs': 'kilogram', 'lbs': 'pound', 'milligram': 'milligram', 'oz': 'ounce', 'gm': 'gram', 'g': 'gram', 'micrograms': 'microgram', 'gsm': 'gram', 'pound': 'pound', 'ton': 'ton', 'mg': 'milligram', 'gram': 'gram', 'kg': 'kilogram', 'microgram': 'microgram', 'µg': 'microgram', 'grams': 'gram', 'kilograms': 'kilogram', 'gms': 'gram', 'kilogram': 'kilogram', 'ounces': 'ounce', 'tons': 'ton', 'lb': 'pound', 'cm': 'centimetre', 'cms': 'centimetre', 'yard': 'yard', 'yd': 'yard', 'in': 'inch', 'centimetre': 'centimetre', 'metre': 'metre', 'feet': 'foot', 'foot': 'foot', 'inch': 'inch', 'mm': 'millimetre', 'millimetre': 'millimetre', 'millimeter': 'millimetre', 'centimeter': 'centimetre', 'm': 'metre', 'meter': 'metre', 'ft': 'foot', 'centilitres': 'centilitre', "''s": 'inch', '"s': 'inch', "'s": 'foot', 'cubic inchs': 'cubic inch', 'fl ozs': 'fluid ounce', 'milliliters': 'millilitre', 'cls': 'centilitre', 'liters': 'litre', 'centiliters': 'centilitre', 'mls': 'millilitre', 'cubic foots': 'cubic foot', 'cu ins': 'cubic inch', 'decilitres': 'decilitre', 'deciliters': 'decilitre', 'ls': 'litre', 'millilitres': 'millilitre', 'litres': 'litre', 'cu fts': 'cubic foot', 'microlitres': 'microlitre', 'microliters': 'microlitre', 'dls': 'decilitre', 'µls': 'microgram', 'fls': 'fluid ounce', 'kws': 'kilowatt', 'vs': 'volt', 'kvs': 'kilovolt', 'mvs': 'millivolt', 'ozs': 'ounce', 'gs': 'gram', 'gsms': 'gram', 'mgs': 'milligram', 'µgs': 'microgram', 'yards': 'yard', 'yds': 'yard', 'ins': 'inch', 'centimetres': 'centimetre', 'metres': 'metre', 'feets': 'foot', 'foots': 'foot', 'inches': 'inch', 'mms': 'millimetre', 'millimetres': 'millimetre', 'millimeters': 'millimetre', 'centimeters': 'centimetre', 'ms': 'metre', 'meters': 'metre', 'fts': 'foot',"ws":"watt","fluid":"fluid ounce",}




import re

def find_keywords(text, possible):
    # Join the list into a regex pattern, using '|' to represent 'or' between words, and add an optional 's' for plural forms
    word_pattern = '|'.join([f"{re.escape(word)}s?" for word in possible])

    # Regex pattern to match any characters before the first number,
    # then capture a number followed by a keyword, allowing for multiple such matches
    pattern = fr'(\d+(?:\.\d+)?)\s{{0,2}}({word_pattern})'

    # Search for all matches of number + word
    full_matches = re.findall(pattern, text)

    # Format the output to combine the number and the word
    output = [f"{num} {word}" for num, word in full_matches]

    return output


# text = 'I have 10.5  apples, 20banana al, and 4.54cherry259 ml'
# possible = ["apples", 'banana al', 'cherry','ml']
# find_keywords(text,possible)

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt




# Function to calculate the center point of a bounding box
def calculate_center_point(box):
    x1, y1, x2, y2 = box  # Extract the coordinates
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y


def find_centres(bounding_boxes):


    # Iterate over each bounding box
    for i in range(len(bounding_boxes)):
        coordinates = bounding_boxes[i]["coordinates"][0]  # Get the coordinates list
        center = calculate_center_point(coordinates)  # Calculate the center
        bounding_boxes[i]["center"]=center











# Load the YOLOv8 model with your best.pt weights
model = YOLO('best (1).pt')

def infer():
# Perform inference on an image
  results = model('downloaded.jpg')
  result=results[0]
  # Access the results
  # for result in results:
      # Extract the image with bounding boxes drawn
  annotated_image = result.plot()  # This returns the annotated image as a numpy array

  # Display the image with matplotlib
  # plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
  # plt.axis('off')  # Hide axes
  # plt.show()

  # Save the annotated image manually
  output_path = 'annotated.jpg'
  cv2.imwrite(output_path, annotated_image)
  ret=[]
  # To access the individual detected results (bounding boxes, confidence scores, etc.)
  boxes = result.boxes  # Contains bounding boxes
  for box in boxes:
      ret.append({"coordinates":box.xyxy,"confidence":box.conf,"class":box.cls})
      # Print box details (coordinates, confidence, and class id)
      # print(f"coordinates: {box.xyxy}, confidence: {box.conf}, class: {box.cls}")
  find_centres(ret)
  return ret


import math

# Function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


def find(path,selected_quantity):
    # image_url = row["image_link"]
    # image_path = "downloaded.jpg"  # You can change the filename if needed
    # download_image(image_url, image_path)

    typ = selected_quantity
    print("required=",typ)
    if typ not in ["height", "depth", "width"]:
        extracted_text = extract_text(path)
        # print(extracted_text)
        poss = orgasm[typ]
        final_ot=[]
        for item in extracted_text:
            item = item.lower()
            ot = find_keywords(item,poss)
            # print(ot)
            final_ot.extend(ot)
        if len(final_ot):
            final_ot.sort(reverse=True)
            answer=final_ot[0]
            answer=answer.split()
            # results = collection.query(
            # query_texts=[answer[1]],
            # n_results=1)
            unit=mapp[answer[1]]
            answer=answer[0]+" "+unit
            print("found=",answer)
            F_AA=answer
        else:
          F_AA="10 inch"
          # can use some calculations? or simply give all text to nlp model
    else:
      text_wbb = extract_wit_position()
      line_info = infer()

      if typ=="height":
          req=[linee for linee in line_info if linee["class"][0]==2]
          if len(req):
              pot_ans=[]
              for lii in req:
                    mi=float("inf")
                    my=""
                    for tt in text_wbb:
                        D=calculate_distance(tt["center"], lii["center"])
                        if D<mi:
                            mi=D
                            my=tt["text"]
                    pot_ans.append(my)
              extracted_text=pot_ans
              # print(extracted_text)
              poss = orgasm[typ]
              final_ot=[]
              for item in extracted_text:
                  item = item.lower()
                  ot = find_keywords(item,poss)
                  if not len(ot):
                      pattern = r"\d+(?:\.\d+)?['\"]{1,2}"

                      # Find all matches
                      matches = re.findall(pattern, item)

                      # Add a space before the quotes in the final output
                      ot= [re.sub(r"(['\"]{1,2})", r" \1", match) for match in matches]
                  final_ot.extend(ot)
              if len(final_ot):
                  spaced=[i.split() for i in final_ot]
                  sorted_lt= sorted(spaced, key=lambda x: float(x[0]), reverse=True)
                  answer=sorted_lt[0]
                  unit=mapp[answer[1]]
                  answer=str(answer[0])+" "+unit
                  print("found=",answer)
                  F_AA=answer
              else:
                F_AA=""
                #find 90x90 ones in line_wbb from extracted_text



          else:
              # no lines , check for only  texts
              F_AA=answer


      elif typ=="width":
          req=[linee for linee in line_info if linee["class"][0]==0]
          if len(req):
              pot_ans=[]
              # big horizontal lines give width (do this)
              for lii in req:
                    mi=float("inf")
                    my=""
                    for tt in text_wbb:
                        D=calculate_distance(tt["center"], lii["center"])
                        if D<mi:
                            mi=D
                            my=tt["text"]
                    pot_ans.append(my)
              extracted_text=pot_ans
              # print(extracted_text)
              poss = orgasm[typ]
              final_ot=[]
              for item in extracted_text:
                  item = item.lower()
                  ot = find_keywords(item,poss)
                  if not len(ot):
                      pattern = r"\d+(?:\.\d+)?['\"]{1,2}"

                      # Find all matches
                      matches = re.findall(pattern, item)

                      # Add a space before the quotes in the final output
                      ot= [re.sub(r"(['\"]{1,2})", r" \1", match) for match in matches]
                  final_ot.extend(ot)
              if len(final_ot):
                  spaced=[i.split() for i in final_ot]
                  sorted_lt= sorted(spaced, key=lambda x: float(x[0]), reverse=True)
                  answer=sorted_lt[0]
                  unit=mapp[answer[1]]
                  answer=str(answer[0])+" "+unit
                  print("found=",answer)
                  F_AA=answer
              else:

                # can use some calculations?
                F_AA=""



          else:
            req=[linee for linee in line_info if linee["class"][0]==1]
            if len(req):
                pot_ans=[]
                for lii in req:
                      mi=float("inf")
                      my=""
                      for tt in text_wbb:
                          D=calculate_distance(tt["center"], lii["center"])
                          if D<mi:
                              mi=D
                              my=tt["text"]
                      pot_ans.append(my)
                extracted_text=pot_ans
                # print(extracted_text)
                poss = orgasm[typ]
                final_ot=[]
                for item in extracted_text:
                    item = item.lower()
                    ot = find_keywords(item,poss)
                    if not len(ot):
                        pattern = r"\d+(?:\.\d+)?['\"]{1,2}"

                        # Find all matches
                        matches = re.findall(pattern, item)

                        # Add a space before the quotes in the final output
                        ot= [re.sub(r"(['\"]{1,2})", r" \1", match) for match in matches]
                    final_ot.extend(ot)
                if len(final_ot):
                    spaced=[i.split() for i in final_ot]
                    sorted_lt= sorted(spaced, key=lambda x: float(x[0]),reverse=True)
                    answer=sorted_lt[0]
                    unit=mapp[answer[1]]
                    answer=str(answer[0])+" "+unit
                    print("found=",answer)
                    F_AA=answer
                else:
                  F_AA=""
                  # can use some calculations?

      else:
          req=[linee for linee in line_info if linee["class"][0]==1]
          if len(req):
              pot_ans=[]
              for lii in req:
                    mi=float("inf")
                    my=""
                    for tt in text_wbb:
                        D=calculate_distance(tt["center"], lii["center"])
                        if D<mi:
                            mi=D
                            my=tt["text"]
                    pot_ans.append(my)
              extracted_text=pot_ans
              # print(extracted_text)
              poss = orgasm[typ]
              final_ot=[]
              for item in extracted_text:
                  item = item.lower()
                  ot = find_keywords(item,poss)
                  if not len(ot):
                      pattern = r"\d+(?:\.\d+)?['\"]{1,2}"

                      # Find all matches
                      matches = re.findall(pattern, item)

                      # Add a space before the quotes in the final output
                      ot= [re.sub(r"(['\"]{1,2})", r" \1", match) for match in matches]
                  final_ot.extend(ot)
              if len(final_ot):
                  spaced=[i.split() for i in final_ot]
                  sorted_lt= sorted(spaced, key=lambda x: float(x[0]))
                  answer=sorted_lt[0]
                  unit=mapp[answer[1]]
                  answer=str(answer[0])+" "+unit
                  print("found=",answer)
                  F_AA=answer
              else:
                F_AA=""
                # can use some calculations?



          else:
              req=[linee for linee in line_info if linee["class"][0]==0]
              if len(req):
                  pot_ans=[]
                  # big horizontal lines give width (do this)
                  for lii in req:
                        mi=float("inf")
                        my=""
                        for tt in text_wbb:
                            D=calculate_distance(tt["center"], lii["center"])
                            if D<mi:
                                mi=D
                                my=tt["text"]
                        pot_ans.append(my)
                  extracted_text=pot_ans
                  # print(extracted_text)
                  poss = orgasm[typ]
                  final_ot=[]
                  for item in extracted_text:
                      item = item.lower()
                      ot = find_keywords(item,poss)
                      if not len(ot):
                        pattern = r"\d+(?:\.\d+)?['\"]{1,2}"

                        # Find all matches
                        matches = re.findall(pattern, item)

                        # Add a space before the quotes in the final output
                        ot= [re.sub(r"(['\"]{1,2})", r" \1", match) for match in matches]
                      final_ot.extend(ot)
                  if len(final_ot):
                      spaced=[i.split() for i in final_ot]
                      #sorted_lt= sorted(spaced, key=lambda x: (float[0]))
                      sorted_lt = sorted(spaced, key=lambda x: float(x[0]))

                      answer=sorted_lt[0]
                      unit=mapp[answer[1]]
                      answer=str(answer[0])+" "+unit
                      print("found=",answer)
                      F_AA=answer
                  else:

                    # can use some calculations?
                    F_AA=""



              else:
                F_AA=""
    return F_AA
import os


# Title of the app
st.title('Image Link Input with Quantity Selector')

# Input an image URL
image_url = st.text_input("Enter the image URL")

# Dropdown menu for quantity selection
quantity_options = [
    "height", "width", "depth", "item_weight",
    "maximum_weight_recommendation", "voltage", 
    "wattage", "item_volume"
]
selected_quantity = st.selectbox("Select a quantity", quantity_options)

# Show button
if st.button('Show Quantity'):
    download_image(image_url, "downloaded.jpg")
    if image_url is not None:
        # Display the uploaded image
        image = Image.open("downloaded.jpg")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Display some placeholder text
        st.write(f"You selected: {selected_quantity}")
        


        F_AA=find('downloaded.jpg',selected_quantity)
        if os.path.exists("annotated.jpg"):
            saved_image = Image.open("annotated.jpg")
            st.image(saved_image, caption="Saved Image", use_column_width=True)

        # Here you can add your code to process the selected quantity
        # and return a result, for now, it's just a placeholder:
        result_text = f"{selected_quantity}:{F_AA}."
        st.write(result_text)
        
    else:
        st.write("Please upload an image.")
