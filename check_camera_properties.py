#!/usr/bin/env python3
"""
Camera Properties Checker

This script connects to a specified camera and lists the values of common
OpenCV properties (like resolution, FPS, exposure, etc.).

This helps in understanding what controls are available for your specific
camera through OpenCV.
"""

import cv2
import argparse

def get_property_name(prop_id, cv2_module):
    """Gets the string name of a cv2.CAP_PROP_* constant."""
    for name in dir(cv2_module):
        if name.startswith('CAP_PROP_') and getattr(cv2_module, name) == prop_id:
            return name
    return None

def check_camera_properties(camera_source=0):
    """
    Connects to a camera and prints its supported properties.
    """
    print(f"🎥 Checking properties for camera source: {camera_source}")
    cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        print(f"❌ Error: Cannot open camera source {camera_source}")
        return

    print("✅ Camera opened successfully. Reading properties...\n")

    # List of common properties to check
    # Note: Not all properties are supported by all cameras or backends.
    properties_to_check = [
        cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        cv2.CAP_PROP_FPS,
        cv2.CAP_PROP_BRIGHTNESS,
        cv2.CAP_PROP_CONTRAST,
        cv2.CAP_PROP_SATURATION,
        cv2.CAP_PROP_HUE,
        cv2.CAP_PROP_GAIN,
        cv2.CAP_PROP_EXPOSURE,
        cv2.CAP_PROP_FOCUS,
        cv2.CAP_PROP_AUTOFOCUS,
        cv2.CAP_PROP_AUTO_EXPOSURE,
        cv2.CAP_PROP_ZOOM,
        cv2.CAP_PROP_BACKEND,
        cv2.CAP_PROP_FOURCC, # Codec
    ]

    print(f"{'Property Name':<25} | {'Value'}")
    print("-" * 40)

    for prop_id in properties_to_check:
        value = cap.get(prop_id)
        prop_name = get_property_name(prop_id, cv2)
        
        if prop_name:
            if prop_name == "CAP_PROP_FOURCC":
                # Decode FOURCC value into characters
                try:
                    fourcc_str = "".join([chr((int(value) >> 8 * i) & 0xFF) for i in range(4)])
                    print(f"{prop_name:<25} | {value} ('{fourcc_str}')")
                except Exception:
                    print(f"{prop_name:<25} | {value}")
            else:
                print(f"{prop_name:<25} | {value}")

    print("\nNotes:")
    print("- A value of 0 or -1 often means the property is not supported or applicable.")
    print("- Brightness/Contrast/etc. values are often in a range (e.g., 0-255 or -1 to 1), but this varies.")
    
    # Attempt to set a property (e.g., resolution) to demonstrate
    print("\n🔧 Attempting to set resolution to 1280x720...")
    
    # Get original values
    original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Set new values
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Read back the values to see if they were accepted
    new_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    new_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"Original resolution: {int(original_width)}x{int(original_height)}")
    print(f"Attempted to set:    1280x720")
    print(f"Actual new resolution: {int(new_width)}x{int(new_height)}")
    
    if int(new_width) == 1280 and int(new_height) == 720:
        print("✅ Resolution successfully changed.")
    else:
        print("⚠️  Resolution was not changed. The camera may not support this resolution or the change was ignored by the driver.")


    cap.release()
    print("\n✅ Camera released.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check OpenCV camera properties.")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="The camera source index (e.g., 0 for default).",
    )
    args = parser.parse_args()
    
    check_camera_properties(args.camera) 