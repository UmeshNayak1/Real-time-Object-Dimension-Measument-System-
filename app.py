# import os
# os.environ["TORCH_DISABLE_WATCH"] = "1"

# import requests
# import streamlit as st
# import numpy as np
# import cv2
# from PIL import Image
# from io import BytesIO
# from overkillpone import process_image
# import database  # Import your database helper

# # Initialize DB table
# database.initialize_table()

# st.set_page_config(page_title="YOLO + Depth Estimation App", layout="wide")
# st.title("📸 YOLO + Depth Estimation (IP Webcam and Local Upload)")

# # Sidebar Navigation
# page = st.sidebar.selectbox("Choose Page", ["Detect Objects", "View Saved Results"])

# if page == "Detect Objects":
#     option = st.radio("Choose Input Source:", ["Real-Time (IP Webcam)", "Local Image Upload"])

#     import time

    # if option == "Real-Time (IP Webcam)":
    #     st.header("Live Preview from IP Webcam")
    #     ip_url = st.text_input("Enter IP Webcam Snapshot URL", value="http://192.168.29.79:8001/shot.jpg")
    #     preview = st.checkbox("🔁 Show Live Preview")
    #     capture = st.button("📸 Capture Image")
    #     frame_placeholder = st.empty()

    #     if ip_url:
    #         if preview and not capture:
    #             while True:
    #                 try:
    #                     response = requests.get(ip_url, timeout=5)
    #                     if response.status_code == 200:
    #                         img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    #                         frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    #                         if frame is None:
    #                             st.error("⚠️ Failed to decode image. Check IP Webcam feed.")
    #                             break
    #                         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #                         frame_placeholder.image(frame, channels="RGB", use_container_width=True)
    #                         time.sleep(1)
    #                     else:
    #                         st.error("⚠️ Could not reach IP Webcam URL.")
    #                         break
    #                 except Exception as e:
    #                     st.error(f"Failed to load preview: {e}")
    #                     break

    #         if capture:
    #             try:
    #                 response = requests.get(ip_url, timeout=5)
    #                 if response.status_code == 200:
    #                     img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    #                     frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    #                     if frame is None:
    #                         st.error("⚠️ Failed to decode image.")
    #                     else:
    #                         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #                         img_pil = Image.fromarray(frame)

    #                         st.image(img_pil, caption="Captured Frame", use_container_width=True)

    #                         with st.spinner("Processing..."):
    #                             output_image, depths, widths, heights, classes = process_image(img_pil)

    #                         st.image(output_image, caption="Detection Result", use_container_width=True)
    #                         st.success("Processing Done ✅")

    #                         if classes:
    #                             detections_list = []
    #                             for i, cls in enumerate(classes):
    #                                 result = f"{cls}: Depth={depths[i]:.2f}m, Width={widths[i]:.2f}m, Height={heights[i]:.2f}m"
    #                                 st.write(result)
    #                                 detections_list.append(result)

    #                             st.session_state["output_image"] = output_image
    #                             st.session_state["detections_list"] = detections_list
    #                 else:
    #                     st.error("⚠️ Could not fetch image from IP Webcam.")
    #             except Exception as e:
    #                 st.error(f"Capture failed: {e}")

#     elif option == "Local Image Upload":
#         st.header("Upload an Image from Local Storage")
#         uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#         if uploaded_file is not None:
#             img_pil = Image.open(uploaded_file).convert("RGB")
#             st.image(img_pil, caption="Uploaded Image", use_container_width=True)

#             if st.button("Process Uploaded Image", key="process_uploaded_button"):
#                 with st.spinner('Processing...'):
#                     output_image, depths, widths, heights, classes = process_image(img_pil)

#                 st.image(output_image, caption="Detection Result", use_container_width=True)
#                 st.success("Processing Done ✅")

#                 if classes:
#                     detections_list = []
#                     for i, cls in enumerate(classes):
#                         result = f"{cls}: Depth={depths[i]:.2f}m, Width={widths[i]:.2f}m, Height={heights[i]:.2f}m"
#                         st.write(result)
#                         detections_list.append(result)

#                     st.session_state["output_image"] = output_image
#                     st.session_state["detections_list"] = detections_list

#     if "output_image" in st.session_state and "detections_list" in st.session_state:
#         if st.button("Save Result to Database", key="save_result_button"):
#             database.save_result_to_db(st.session_state["output_image"], st.session_state["detections_list"])
#             st.success("Result Saved to Database ✅")

# elif page == "View Saved Results":
#     st.header("📂 Saved Detection Results")
#     saved_results = database.load_saved_results()

#     if saved_results:
#         for row in saved_results:
#             img_bytes = row['image']
#             detections = row['detections']
#             timestamp = row['timestamp']
#             result_id = row['id']

#             img_pil = Image.open(BytesIO(img_bytes))
#             timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else "Unknown Time"

#             st.image(img_pil, caption=f"Saved on {timestamp_str}", use_container_width=True)
#             st.write("Detections:")
#             st.json(eval(detections))

#             if st.button(f"Delete This Result {result_id}", key=f"delete_{result_id}"):
#                 database.delete_result(result_id)
#                 st.success("Deleted! Please refresh page.")
#     else:
#         st.info("No saved results yet.")


# app.py

# import os
# os.environ["TORCH_DISABLE_WATCH"] = "1"


# import streamlit as st
# import requests
# import numpy as np
# from PIL import Image
# from io import BytesIO
# from overkillpone import process_image
# import database

# database.initialize_table()

# st.set_page_config(page_title="YOLO + Depth Estimation App", layout="wide")
# st.title("📸 YOLO + Depth Estimation - IP Webcam, DroidCam USB (Flask), Local Upload")

# page = st.sidebar.selectbox("Choose Page", ["Detect Objects", "View Saved Results"])

# if page == "Detect Objects":
#     option = st.radio("Choose Input Source:", [
#         "IP Webcam",
#         "DroidCam USB (via Flask)",
#         "Local Image Upload"
#     ])

    # if option == "IP Webcam":
    #     st.header("📱 IP Webcam")
    #     ip_url = st.text_input("Enter IP Webcam Snapshot URL", value="http://192.168.29.79:8001/shot.jpg")
    #     if st.button("📸 Capture from IP Webcam"):
    #         try:
    #             response = requests.get(ip_url, timeout=5)
    #             if response.status_code == 200:
    #                 img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    #                 frame = Image.open(BytesIO(img_array)).convert("RGB")
    #                 st.image(frame, caption="Captured Frame", use_container_width=True)

    #                 with st.spinner("Processing..."):
    #                     output_image, depths, widths, heights, classes = process_image(frame)

    #                 st.image(output_image, caption="Detection Result", use_container_width=True)
    #                 st.success("Processing Done ✅")

    #                 if classes:
    #                     detections_list = []
    #                     for i, cls in enumerate(classes):
    #                         st.write(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")
    #                         detections_list.append(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")

    #                     st.session_state["output_image"] = output_image
    #                     st.session_state["detections_list"] = detections_list
    #             else:
    #                 st.error("❌ Could not reach IP Webcam URL.")
    #         except Exception as e:
    #             st.error(f"❌ Error: {e}")

#     elif option == "DroidCam USB (via Flask)":
#         st.header("📷 Capture from DroidCam USB via Flask")
#         if st.button("📸 Capture from USB Camera"):
#             try:
#                 response = requests.get("http://localhost:5000/capture", timeout=5)
#                 if response.status_code == 200:
#                     img_bytes = BytesIO(response.content)
#                     img_pil = Image.open(img_bytes).convert("RGB")
#                     st.image(img_pil, caption="Captured Frame", use_container_width=True)

#                     with st.spinner("Processing..."):
#                         output_image, depths, widths, heights, classes = process_image(img_pil)

#                     st.image(output_image, caption="Detection Result", use_container_width=True)
#                     st.success("Processing Done ✅")

#                     if classes:
#                         detections_list = []
#                         for i, cls in enumerate(classes):
#                             st.write(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")
#                             detections_list.append(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")

#                         st.session_state["output_image"] = output_image
#                         st.session_state["detections_list"] = detections_list
#                 else:
#                     st.error("❌ Could not capture from Flask backend.")
#             except Exception as e:
#                 st.error(f"❌ Error contacting Flask backend: {e}")

#     elif option == "Local Image Upload":
#         st.header("📁 Upload Local Image")
#         file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
#         if file:
#             img = Image.open(file).convert("RGB")
#             st.image(img, caption="Uploaded Image", use_container_width=True)
#             if st.button("Process Uploaded Image"):
#                 with st.spinner("Processing..."):
#                     output_image, depths, widths, heights, classes = process_image(img)

#                 st.image(output_image, caption="Detection Result", use_container_width=True)
#                 st.success("Processing Done ✅")

#                 if classes:
#                     detections_list = []
#                     for i, cls in enumerate(classes):
#                         st.write(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")
#                         detections_list.append(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")

#                     st.session_state["output_image"] = output_image
#                     st.session_state["detections_list"] = detections_list

#     if "output_image" in st.session_state and "detections_list" in st.session_state:
#         if st.button("💾 Save to Database"):
#             database.save_result_to_db(
#                 st.session_state["output_image"],
#                 st.session_state["detections_list"]
#             )
#             st.success("✅ Saved to database.")

# elif page == "View Saved Results":
#     st.header("📂 Saved Results")
#     results = database.load_saved_results()

#     if results:
#         for row in results:
#             img_bytes = row['image']
#             detections = row['detections']
#             timestamp = row['timestamp']
#             result_id = row['id']

#             img_pil = Image.open(BytesIO(img_bytes))
#             time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else "Unknown"

#             st.image(img_pil, caption=f"Saved on {time_str}", use_container_width=True)
#             st.write("Detections:")
#             st.json(eval(detections))

#             if st.button(f"🗑️ Delete {result_id}", key=f"delete_{result_id}"):
#                 database.delete_result(result_id)
#                 st.success("Deleted. Refresh to update.")
#     else:
#         st.info("No saved results yet.")



#   perfect code but no capturwe button 

# import os
# os.environ["TORCH_DISABLE_WATCH"] = "1"

# import streamlit as st
# import requests
# import numpy as np
# from PIL import Image
# from io import BytesIO
# from overkillpone import process_image
# import database

# database.initialize_table()

# st.set_page_config(page_title="YOLO + Depth Estimation App", layout="wide")
# st.title("📸 YOLO + Depth Estimation - IP Webcam, DroidCam USB (Flask), Local Upload")

# page = st.sidebar.selectbox("Choose Page", ["Detect Objects", "View Saved Results"])

# if page == "Detect Objects":
#     option = st.radio("Choose Input Source:", [
#         "IP Webcam",
#         "DroidCam USB (via Flask)",
#         "Local Image Upload"
#     ])

    # if option == "IP Webcam":
    #     st.header("📱 IP Webcam")
    #     ip_url = st.text_input("Enter IP Webcam Snapshot URL", value="http://192.168.29.79:8001/shot.jpg")

    #     if st.button("🔁 Show Live Preview", key="ipwebcam_preview"):
    #         stframe = st.empty()
    #         while True:
    #             try:
    #                 response = requests.get(ip_url, timeout=2)
    #                 if response.status_code == 200:
    #                     img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    #                     frame = Image.open(BytesIO(img_array)).convert("RGB")
    #                     stframe.image(frame, caption="Live Preview", use_container_width=True)
    #                 else:
    #                     st.error("❌ IP Webcam feed not reachable.")
    #                     break
    #             except Exception:
    #                 break

    #     if st.button("📸 Capture from IP Webcam", key="ipwebcam_capture"):
    #         try:
    #             response = requests.get(ip_url, timeout=5)
    #             if response.status_code == 200:
    #                 img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    #                 frame = Image.open(BytesIO(img_array)).convert("RGB")
    #                 st.image(frame, caption="Captured Frame", use_container_width=True)

    #                 with st.spinner("Processing..."):
    #                     output_image, depths, widths, heights, classes = process_image(frame)

    #                 st.image(output_image, caption="Detection Result", use_container_width=True)
    #                 st.success("Processing Done ✅")

    #                 if classes:
    #                     detections_list = []
    #                     for i, cls in enumerate(classes):
    #                         st.write(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")
    #                         detections_list.append(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")

    #                     st.session_state["output_image"] = output_image
    #                     st.session_state["detections_list"] = detections_list
    #             else:
    #                 st.error("❌ Could not reach IP Webcam URL.")
    #         except Exception as e:
    #             st.error(f"❌ Error: {e}")

#     elif option == "DroidCam USB (via Flask)":
#         st.header("📷 DroidCam USB (Flask backend)")

#         flask_host = st.text_input("Enter Flask server URL", value="http://localhost:5000")
#         preview_url = f"{flask_host}/preview"
#         capture_url = f"{flask_host}/capture"

#         if st.button("🔁 Show Live Preview", key="droidcam_preview"):
#             stframe = st.empty()
#             while True:
#                 try:
#                     response = requests.get(preview_url, timeout=2)
#                     if response.status_code == 200:
#                         img_bytes = BytesIO(response.content)
#                         frame = Image.open(img_bytes).convert("RGB")
#                         stframe.image(frame, caption="Live Preview", use_container_width=True)
#                     else:
#                         st.error("❌ Could not fetch from Flask preview.")
#                         break
#                 except Exception:
#                     break

#         if st.button("📸 Capture from DroidCam", key="droidcam_capture"):
#             try:
#                 response = requests.get(capture_url, timeout=5)
#                 if response.status_code == 200:
#                     img_bytes = BytesIO(response.content)
#                     frame = Image.open(img_bytes).convert("RGB")
#                     st.image(frame, caption="Captured Frame", use_container_width=True)

#                     with st.spinner("Processing..."):
#                         output_image, depths, widths, heights, classes = process_image(frame)

#                     st.image(output_image, caption="Detection Result", use_container_width=True)
#                     st.success("Processing Done ✅")

#                     if classes:
#                         detections_list = []
#                         for i, cls in enumerate(classes):
#                             st.write(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")
#                             detections_list.append(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")

#                         st.session_state["output_image"] = output_image
#                         st.session_state["detections_list"] = detections_list
#                 else:
#                     st.error("❌ Could not capture from Flask backend.")
#             except Exception as e:
#                 st.error(f"❌ Error contacting Flask backend: {e}")

#     elif option == "Local Image Upload":
#         st.header("📁 Upload Local Image")
#         file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
#         if file:
#             img = Image.open(file).convert("RGB")
#             st.image(img, caption="Uploaded Image", use_container_width=True)
#             if st.button("Process Uploaded Image"):
#                 with st.spinner("Processing..."):
#                     output_image, depths, widths, heights, classes = process_image(img)

#                 st.image(output_image, caption="Detection Result", use_container_width=True)
#                 st.success("Processing Done ✅")

#                 if classes:
#                     detections_list = []
#                     for i, cls in enumerate(classes):
#                         st.write(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")
#                         detections_list.append(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")

#                     st.session_state["output_image"] = output_image
#                     st.session_state["detections_list"] = detections_list

#     if "output_image" in st.session_state and "detections_list" in st.session_state:
#         if st.button("💾 Save to Database"):
#             database.save_result_to_db(
#                 st.session_state["output_image"],
#                 st.session_state["detections_list"]
#             )
#             st.success("✅ Saved to database.")

# elif page == "View Saved Results":
#     st.header("📂 Saved Results")
#     results = database.load_saved_results()

#     if results:
#         for row in results:
#             img_bytes = row['image']
#             detections = row['detections']
#             timestamp = row['timestamp']
#             result_id = row['id']

#             img_pil = Image.open(BytesIO(img_bytes))
#             time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else "Unknown"

#             st.image(img_pil, caption=f"Saved on {time_str}", use_container_width=True)
#             st.write("Detections:")
#             st.json(eval(detections))

#             if st.button(f"🗑️ Delete {result_id}", key=f"delete_{result_id}"):
#                 database.delete_result(result_id)
#                 st.success("Deleted. Refresh to update.")
#     else:
#         st.info("No saved results yet.")
















import os
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import streamlit as st
from overkillpone import process_image  # Assuming this is the module for your YOLO model
import database

database.initialize_table()

st.set_page_config(page_title="YOLO + Depth Estimation App", layout="wide")
st.title("📸 YOLO + Depth Estimation - IP Webcam, DroidCam USB (Flask), Local Upload")

page = st.sidebar.selectbox("Choose Page", ["Detect Objects", "View Saved Results"])

if page == "Detect Objects":
    option = st.radio("Choose Input Source:", [
        "IP Webcam",
        "DroidCam USB (via Flask)",
        "Local Image Upload"
    ])
    

    
    # if option == "IP Webcam":
    #     st.header("📱 IP Webcam")
    #     ip_url = st.text_input("Enter IP Webcam Snapshot URL", value="http://192.168.29.79:8001/shot.jpg")
        
    #     # Display live preview of IP Webcam
    #     st.image(ip_url, caption="IP Webcam Feed")

    #     # Display live feed from IP Webcam and process frame on button click
    #     if st.button("Start Live Preview"):
    #         try:
    #             # Fetch live feed continuously (we'll get latest snapshot every few seconds)
    #             img_response = requests.get(ip_url, timeout=5)
    #             if img_response.status_code == 200:
    #                 img_array = np.asarray(bytearray(img_response.content), dtype=np.uint8)
    #                 frame = Image.open(BytesIO(img_array)).convert("RGB")
    #                 st.image(frame, caption="Live Camera Feed", use_container_width=True)

    #                 # Capture button
    #                 if st.button("Capture and Process"):
    #                     with st.spinner("Processing..."):
    #                         output_image, depths, widths, heights, classes = process_image(frame)
    #                     st.image(output_image, caption="Detection Result", use_container_width=True)
    #                     st.success("Processing Done ✅")

    #                     # Display results
    #                     if classes:
    #                         detections_list = []
    #                         for i, cls in enumerate(classes):
    #                             st.write(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")
    #                             detections_list.append(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")

    #                         st.session_state["output_image"] = output_image
    #                         st.session_state["detections_list"] = detections_list
    #             else:
    #                 st.error("❌ Could not reach IP Webcam URL.")
    #         except Exception as e:
    #             st.error(f"❌ Error: {e}")
    
    
    if option == "IP Webcam":
        st.header("📱 IP Webcam")
        ip_url = st.text_input("Enter IP Webcam Snapshot URL", value="http://192.168.29.79:8001/shot.jpg")
        if st.button("📸 Capture from IP Webcam"):
            try:
                response = requests.get(ip_url, timeout=5)
                if response.status_code == 200:
                    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                    frame = Image.open(BytesIO(img_array)).convert("RGB")
                    st.image(frame, caption="Captured Frame", use_container_width=True)

                    with st.spinner("Processing..."):
                        output_image, depths, widths, heights, classes = process_image(frame)

                    st.image(output_image, caption="Detection Result", use_container_width=True)
                    st.success("Processing Done ✅")

                    if classes:
                        detections_list = []
                        for i, cls in enumerate(classes):
                            st.write(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")
                            detections_list.append(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")

                        st.session_state["output_image"] = output_image
                        st.session_state["detections_list"] = detections_list
                else:
                    st.error("❌ Could not reach IP Webcam URL.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    

    elif option == "DroidCam USB (via Flask)":
        st.header("📷 Capture from DroidCam USB via Flask")
        st.image("http://localhost:5000/preview", caption="Live Camera Feed")

        # Capture button to process the current frame
        if st.button("Capture and Process"):
            try:
                response = requests.get("http://localhost:5000/capture", timeout=5)
                if response.status_code == 200:
                    img_bytes = BytesIO(response.content)
                    img_pil = Image.open(img_bytes).convert("RGB")
                    st.image(img_pil, caption="Captured Frame", use_container_width=True)

                    with st.spinner("Processing..."):
                        output_image, depths, widths, heights, classes = process_image(img_pil)

                    st.image(output_image, caption="Detection Result", use_container_width=True)
                    st.success("Processing Done ✅")

                    if classes:
                        detections_list = []
                        for i, cls in enumerate(classes):
                            st.write(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")
                            detections_list.append(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")

                        st.session_state["output_image"] = output_image
                        st.session_state["detections_list"] = detections_list
                else:
                    st.error("❌ Could not capture from Flask backend.")
            except Exception as e:
                st.error(f"❌ Error contacting Flask backend: {e}")

    elif option == "Local Image Upload":
        st.header("📁 Upload Local Image")
        file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if file:
            img = Image.open(file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_container_width=True)
            if st.button("Process Uploaded Image"):
                with st.spinner("Processing..."):
                    output_image, depths, widths, heights, classes = process_image(img)

                st.image(output_image, caption="Detection Result", use_container_width=True)
                st.success("Processing Done ✅")

                if classes:
                    detections_list = []
                    for i, cls in enumerate(classes):
                        st.write(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")
                        detections_list.append(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")

                    st.session_state["output_image"] = output_image
                    st.session_state["detections_list"] = detections_list

    if "output_image" in st.session_state and "detections_list" in st.session_state:
        if st.button("💾 Save to Database"):
            database.save_result_to_db(
                st.session_state["output_image"],
                st.session_state["detections_list"]
            )
            st.success("✅ Saved to database.")

elif page == "View Saved Results":
    st.header("📂 Saved Results")
    results = database.load_saved_results()

    if results:
        for row in results:
            img_bytes = row['image']
            detections = row['detections']
            timestamp = row['timestamp']
            result_id = row['id']

            img_pil = Image.open(BytesIO(img_bytes))
            time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else "Unknown"

            st.image(img_pil, caption=f"Saved on {time_str}", use_container_width=True)
            st.write("Detections:")
            st.json(eval(detections))

            if st.button(f"🗑️ Delete {result_id}", key=f"delete_{result_id}"):
                database.delete_result(result_id)
                st.success("Deleted. Refresh to update.")
    else:
        st.info("No saved results yet.")














# import os
# import requests
# import numpy as np
# from PIL import Image
# from io import BytesIO
# import streamlit as st
# import database
# from overkillpone import process_image

# database.initialize_table()

# st.set_page_config(page_title="YOLO + Depth Estimation App", layout="wide")
# st.title("📸 YOLO + Depth Estimation - IP Webcam, DroidCam USB (Flask), Local Upload")

# page = st.sidebar.selectbox("Choose Page", ["Detect Objects", "View Saved Results"])

# if page == "Detect Objects":
#     option = st.radio("Choose Input Source:", [
#         "IP Webcam",
#         "DroidCam USB (via Flask)",
#         "Local Image Upload"
#     ])

#     if option == "IP Webcam":
#         st.header("📱 IP Webcam")
#         ip_url = st.text_input("Enter IP Webcam Video Feed URL", value="http://192.168.29.79:8001/video")

#         if st.button("📸 Start Live Preview from IP Webcam"):
#             try:
#                 # Stream the video feed using the IP Webcam URL
#                 video_feed = requests.get(ip_url, stream=True, timeout=5)
#                 if video_feed.status_code == 200:
#                     st.image(video_feed.raw, caption="Live Feed", use_container_width=True)

#                     # When "Capture" is clicked, get a single frame from the feed
#                     if st.button("Capture"):
#                         img_array = np.asarray(bytearray(video_feed.raw.read()), dtype=np.uint8)
#                         frame = Image.open(BytesIO(img_array)).convert("RGB")

#                         with st.spinner("Processing..."):
#                             output_image, depths, widths, heights, classes = process_image(frame)

#                         st.image(output_image, caption="Detection Result", use_container_width=True)
#                         st.success("Processing Done ✅")

#                         if classes:
#                             detections_list = []
#                             for i, cls in enumerate(classes):
#                                 st.write(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")
#                                 detections_list.append(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")

#                             st.session_state["output_image"] = output_image
#                             st.session_state["detections_list"] = detections_list
#                 else:
#                     st.error("❌ Could not connect to IP Webcam feed.")
#             except Exception as e:
#                 st.error(f"❌ Error: {e}")

#     elif option == "DroidCam USB (via Flask)":
#         st.header("📷 Capture from DroidCam USB via Flask")
#         if st.button("📸 Capture from USB Camera"):
#             try:
#                 response = requests.get("http://localhost:5000/capture", timeout=5)
#                 if response.status_code == 200:
#                     img_bytes = BytesIO(response.content)
#                     img_pil = Image.open(img_bytes).convert("RGB")
#                     st.image(img_pil, caption="Captured Frame", use_container_width=True)

#                     with st.spinner("Processing..."):
#                         output_image, depths, widths, heights, classes = process_image(img_pil)

#                     st.image(output_image, caption="Detection Result", use_container_width=True)
#                     st.success("Processing Done ✅")

#                     if classes:
#                         detections_list = []
#                         for i, cls in enumerate(classes):
#                             st.write(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")
#                             detections_list.append(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")

#                         st.session_state["output_image"] = output_image
#                         st.session_state["detections_list"] = detections_list
#                 else:
#                     st.error("❌ Could not capture from Flask backend.")
#             except Exception as e:
#                 st.error(f"❌ Error contacting Flask backend: {e}")

#     elif option == "Local Image Upload":
#         st.header("📁 Upload Local Image")
#         file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
#         if file:
#             img = Image.open(file).convert("RGB")
#             st.image(img, caption="Uploaded Image", use_container_width=True)
#             if st.button("Process Uploaded Image"):
#                 with st.spinner("Processing..."):
#                     output_image, depths, widths, heights, classes = process_image(img)

#                 st.image(output_image, caption="Detection Result", use_container_width=True)
#                 st.success("Processing Done ✅")

#                 if classes:
#                     detections_list = []
#                     for i, cls in enumerate(classes):
#                         st.write(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")
#                         detections_list.append(f"{cls}: Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")

#                     st.session_state["output_image"] = output_image
#                     st.session_state["detections_list"] = detections_list

#     if "output_image" in st.session_state and "detections_list" in st.session_state:
#         if st.button("💾 Save to Database"):
#             database.save_result_to_db(
#                 st.session_state["output_image"],
#                 st.session_state["detections_list"]
#             )
#             st.success("✅ Saved to database.")

# elif page == "View Saved Results":
#     st.header("📂 Saved Results")
#     results = database.load_saved_results()

#     if results:
#         for row in results:
#             img_bytes = row['image']
#             detections = row['detections']
#             timestamp = row['timestamp']
#             result_id = row['id']

#             img_pil = Image.open(BytesIO(img_bytes))
#             time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else "Unknown"

#             st.image(img_pil, caption=f"Saved on {time_str}", use_container_width=True)
#             st.write("Detections: ")
#             st.json(eval(detections))

#             if st.button(f"🗑️ Delete {result_id}", key=f"delete_{result_id}"):
#                 database.delete_result(result_id)
#                 st.success("Deleted. Refresh to update.")
#     else:
#         st.info("No saved results yet.")
