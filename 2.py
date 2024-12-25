def run_inference(
    model_path,
    image_path,
    conf_thres=0.25,
    iou_thres=0.45,
    img_size=(640, 640),
    classes=None
):
    """
    Run ONNX model inference on an image.
    """
    # Load ONNX model
    net = cv2.dnn.readNetFromONNX(model_path)
    
    # Read and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Make a copy for drawing
    display_img = img.copy()
    
    original_shape = img.shape
    print(f"Original image shape: {original_shape}")
    
    # Resize and normalize image
    blob = cv2.dnn.blobFromImage(
        img, 
        1/255.0, 
        img_size, 
        swapRB=True, 
        crop=False
    )
    
    # Run inference
    net.setInput(blob)
    outputs = net.forward()
    print(f"\nRaw model output shape: {outputs.shape}")
    
    # Process detections
    if len(outputs.shape) == 3:
        outputs = outputs[0]
    elif len(outputs.shape) == 4:
        outputs = outputs[0]
    
    boxes = []
    scores = []
    class_ids = []
    
    # Extract detections that meet confidence threshold
    for detection in outputs:
        if len(detection) < 5:
            continue
            
        confidence = float(detection[4])
        
        if confidence >= conf_thres:
            if len(detection) < 6:
                continue
                
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            score = float(class_scores[class_id] * confidence)
            
            if score >= conf_thres and (classes is None or class_id in classes):
                # Get normalized box coordinates (x, y, w, h)
                x_center, y_center, width, height = detection[0:4]
                
                # Scale normalized coordinates (0-1) to pixel coordinates
                img_h, img_w = original_shape[:2]
                
                x_center = x_center * img_w
                y_center = y_center * img_h
                width = width * img_w
                height = height * img_h
                
                # Convert from center to top-left corner (x, y) and width, height
                x = x_center - width / 2
                y = y_center - height / 2
                
                # Make sure the bounding box is within image boundaries
                x = max(0, min(x, img_w - 1))
                y = max(0, min(y, img_h - 1))
                width = min(width, img_w - x)
                height = min(height, img_h - y)
                
                print(f"\nDetection found:")
                print(f"  Class: {class_id}")
                print(f"  Confidence: {score:.4f}")
                print(f"  Box (x,y,w,h): {x:.1f}, {y:.1f}, {width:.1f}, {height:.1f}")
                
                boxes.append([x, y, width, height])
                scores.append(score)
                class_ids.append(class_id)
    
    # Apply NMS
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
        print(f"Detections after NMS: {len(indices)}")
        
        # Generate random colors for each class
        np.random.seed(42)
        colors = {i: tuple(map(int, np.random.randint(0, 255, 3))) for i in set(class_ids)}
        
        # Draw detections with different colors for each class
        for i in indices.flatten():  # Flatten in case the indices are returned as a 2D array
            box = boxes[i]
            class_id = class_ids[i]
            score = scores[i]
            color = colors[class_id]
            
            display_img = draw_detection(display_img, box, class_id, score, color)
            
            print(f"\nDrawing detection:")
            print(f"  Class: {class_id}")
            print(f"  Confidence: {score:.4f}")
            print(f"  Box (x,y,w,h): {[f'{v:.1f}' for v in box]}")
    else:
        print("No detections passed the confidence threshold!")
    
    return display_img
