import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from typing import List

def visualize_patch_attention(
    frame_at_toa: np.ndarray,
    bbox_xyxy: tuple,
    epoch: int = 0,
    batch_idx: int = 0,
    save_dir: str = "_visualization",
    is_frame_in_0to1: bool = True,
    attn_map_8x8: torch.Tensor = None,
    plot_all_detections: List[tuple] = None,
    obj_attn_at_toa: torch.Tensor = None
):
    """
    Visualize patch-wise attention over a bounding box on the given image, 
    then save the result to disk.
    
    Args:
        frame_at_toa (np.ndarray):
            The image to draw on. Can be in the range [0,1] or [0,255] 
            and shape (C, H, W) or (H, W, C). This example assumes (C, H, W).
        bbox_xyxy (tuple):
            Bounding box coordinates (x1, y1, x2, y2).
        epoch (int):
            Epoch number (used for saving).
        batch_idx (int):
            Batch index (used for saving).
        save_dir (str):
            Directory to save the visualization.
        is_frame_in_0to1 (bool):
            If True, `frame_at_toa` is assumed to be in [0,1]. Otherwise, [0,255].
        attn_map_8x8 (torch.Tensor):
            A 2D attention map of shape (8, 8) that you want to overlay.
        plot_all_detections (List of tuples):
            List of tuples (frame_idx, bbox_xyxy) to plot. // Default: None
    
    Returns:
        None. The function saves the overlayed image to disk.
    """
    
    # -------------------------------------------------------------------------
    # 1. Preprocess Image
    # -------------------------------------------------------------------------
    # Assume input frame is (C, H, W). If it's (H, W, C), rearrange it.
    if frame_at_toa.shape[0] in [1, 3] and frame_at_toa.shape[-1] not in [1, 3]:
        # shape = (C, H, W), do nothing
        pass
    else:
        # shape = (H, W, C), transpose to (C, H, W)
        frame_at_toa = np.transpose(frame_at_toa, (2, 0, 1))
    
    # Convert float [0,1] -> uint8 [0,255] if needed
    if is_frame_in_0to1:
        frame_at_toa = (frame_at_toa * 255).clip(0, 255).astype(np.uint8)
    else:
        # If it's already in [0,255] just ensure it's uint8
        frame_at_toa = frame_at_toa.astype(np.uint8)
    
    # -------------------------------------------------------------------------
    # 2. Clamp bounding box coordinates
    # -------------------------------------------------------------------------
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    
    # Replace '640' below with the image width/height if needed.
    H, W = frame_at_toa.shape[1], frame_at_toa.shape[2]  # after transpose => (C, H, W)
    x1 = max(0, min(W, x1))
    y1 = max(0, min(H, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))
    
    # -------------------------------------------------------------------------
    # 3. Interpolate the attention map to match the bbox size 
    #    Use nearest interpolation to preserve patch block appearance
    # -------------------------------------------------------------------------
    h = y2 - y1
    w = x2 - x1
    if h <= 0 or w <= 0:
        print("Warning: bounding box has non-positive width/height. Skipping visualization.")
        return
    
    # attn_map_8x8 is shape (8, 8). Expand to (1,1,8,8) for interpolation
    if attn_map_8x8 is not None:
        attn_map_resized = F.interpolate(
            attn_map_8x8.unsqueeze(0).unsqueeze(0), 
            size=(h, w), 
            mode='nearest'
        ).squeeze(0).squeeze(0)  # shape => (h, w)
    
        # Convert to numpy
        attn_map_resized = attn_map_resized.detach().cpu().numpy()
        
        # -------------------------------------------------------------------------
        # 4. Normalize attention map to [0, 1] and apply a color map
        # -------------------------------------------------------------------------
        # Avoid division by zero by adding a small epsilon if needed.
        attn_min, attn_max = attn_map_resized.min(), attn_map_resized.max()
        if abs(attn_max - attn_min) < 1e-10:
            # If the map is all zeros (or constant)
            attn_map_norm = np.zeros_like(attn_map_resized)
        else:
            attn_map_norm = (attn_map_resized - attn_min) / (attn_max - attn_min)
    
        # Apply color map (JET)
        attn_map_colored = cv2.applyColorMap(
            (attn_map_norm * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )  # shape: (h, w, 3)
        
    # -------------------------------------------------------------------------
    # 5. Blend the attention map over the original bounding box region
    # -------------------------------------------------------------------------
        # Copy the original image to overlay
        overlay = frame_at_toa.copy()  # shape: (C, H, W)
        
        # Extract the bounding box region from overlay => shape: (C, h, w)
        bbox_region = overlay[:, y1:y2, x1:x2]
        
        # Convert bounding box region to (h, w, C) for OpenCV blending
        bbox_region_hwc = np.transpose(bbox_region, (1, 2, 0))  # (h, w, C)
        
        # Ensure shapes match for blending
        if bbox_region_hwc.shape[:2] != attn_map_colored.shape[:2]:
            attn_map_colored = cv2.resize(
                attn_map_colored,
                (bbox_region_hwc.shape[1], bbox_region_hwc.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        # Blend: 70% original region + 30% attention
        blended_region = cv2.addWeighted(
            bbox_region_hwc,
            0.7,
            attn_map_colored,
            0.3,
            0
        )
        
        # Put blended region back to overlay
        overlay[:, y1:y2, x1:x2] = np.transpose(blended_region, (2, 0, 1))
    
    else:
        overlay = frame_at_toa.copy()
    # -------------------------------------------------------------------------
    # 6. Draw bounding box and convert to (H, W, C) for saving
    # -------------------------------------------------------------------------
    overlay_hwc = np.transpose(overlay, (1, 2, 0)).copy()  # => (H, W, C)
    
    if plot_all_detections is not None:
        for bbox_xyxy in plot_all_detections:
            
            x1_, y1_, x2_, y2_ = map(int, bbox_xyxy)
            x1_ = max(0, min(W, x1_))
            y1_ = max(0, min(H, y1_))
            x2_ = max(0, min(W, x2_))
            y2_ = max(0, min(H, y2_))
            
            cv2.rectangle(
                img=overlay_hwc,
                pt1=(x1_, y1_),
                pt2=(x2_, y2_),
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
    
    # Draw bounding box in red (BGR=(0,0,255))
    cv2.rectangle(
        img=overlay_hwc,
        pt1=(x1, y1),
        pt2=(x2, y2),
        color=(255, 0, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    
    # After drawing all detection boxes, add attention scores
    if plot_all_detections is not None and obj_attn_at_toa is not None:
        # Ensure obj_attn_at_toa is on CPU and converted to numpy
        attn_scores = obj_attn_at_toa.detach().cpu().numpy()
        attn_scores = attn_scores[0]
        for idx, bbox_xyxy in enumerate(plot_all_detections):
            x1_, y1_, x2_, y2_ = map(int, bbox_xyxy)
            x1_ = max(0, min(W, x1_))
            y1_ = max(0, min(H, y1_))
            
            # Format attention score with 3 decimal places
            attn_score = f"{attn_scores[idx]:.3f}"
            
            # Put text slightly above the bounding box
            text_pos = (x1_, y1_ - 5)
            
            # Add white background for better visibility
            (text_w, text_h), _ = cv2.getTextSize(
                attn_score, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                1
            )
            cv2.rectangle(
                overlay_hwc,
                (text_pos[0] - 2, text_pos[1] - text_h - 2),
                (text_pos[0] + text_w + 2, text_pos[1] + 2),
                (255, 255, 255),
                -1
            )
            
            # Add text with attention score
            cv2.putText(
                overlay_hwc,
                attn_score,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

    # -------------------------------------------------------------------------
    # 7. Save the result (convert from RGB to BGR if needed)
    # -------------------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"epoch_{epoch}_batch_{batch_idx}.jpg")
    
    # Currently overlay_hwc is in (H, W, C) format with channel order = RGB 
    # because we haven't used any BGR-based transformations except in the rectangle. 
    # Typically, you can use cv2.imwrite directly on it, 
    # but if the color appears swapped, do the following:
    overlay_bgr = cv2.cvtColor(overlay_hwc, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, overlay_bgr)


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Example usage
    # -------------------------------------------------------------------------
    # Suppose you have:
    # 1) a dummy (C, H, W) image in [0,1]
    # 2) a dummy 8x8 attention map
    # 3) a bounding box (x1, y1, x2, y2)
    
    # Create a synthetic image: shape (3, 256, 256) with random floats in [0,1]
    img = np.random.rand(3, 256, 256).astype(np.float32)
    
    # Create a random attention map: shape (8, 8)
    attn = torch.rand(8, 8)  # e.g., 8x8 patch-level attention
    
    # Define a bounding box
    bbox = (50, 50, 200, 200)  # (x1, y1, x2, y2)
    
    # Call the function
    visualize_patch_attention(
        frame_at_toa=img,
        attn_map_8x8=attn,
        bbox_xyxy=bbox,
        epoch=0,
        batch_idx=0,
        save_dir="_visualization",
        is_frame_in_0to1=True
    )

    # Add example for attention scores
    plot_all_detections = [(50, 50, 200, 200), (100, 100, 150, 150)]
    obj_attn_at_toa = torch.tensor([0.8, 0.3])  # Example attention scores
    
    visualize_patch_attention(
        frame_at_toa=img,
        attn_map_8x8=attn,
        bbox_xyxy=bbox,
        epoch=0,
        batch_idx=0,
        save_dir="_visualization",
        is_frame_in_0to1=True,
        plot_all_detections=plot_all_detections,
        obj_attn_at_toa=obj_attn_at_toa
    )
