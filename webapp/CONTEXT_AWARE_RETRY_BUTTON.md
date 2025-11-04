# Context-Aware "Choose Another" Button - Feature Documentation

## Overview
The "Choose Another" button in the image preview card is now **context-aware**. It remembers how the user selected their image and provides the appropriate action when clicked.

## The Problem
Previously, the "Choose Another" button always opened the file gallery, even when the user had captured a photo using the camera. This created a confusing user experience where:
- User clicks "Take Photo" → Opens camera
- User captures photo → Preview shows
- User clicks "Choose Another" → Opens file gallery (unexpected!)
- User expected: Camera to reopen for recapture

## The Solution

### Smart Button Behavior
The button now adapts based on the **image source**:

#### When Image from Camera 📷
```
Button Icon: Camera icon (fa-camera)
Button Text: "Retake Photo"
Button Action: Reopens camera for new capture
```

#### When Image from Gallery 🖼️
```
Button Icon: Rotate icon (fa-rotate-right)
Button Text: "Choose Another"
Button Action: Opens file browser for new selection
```

## How It Works

### 1. Source Tracking
```javascript
// Global variable tracks the current image source
let imageSource = null; // Values: 'camera' or 'gallery'
```

### 2. Source Assignment
Every time `showPreview()` is called, it receives and stores the source:

**From Gallery:**
```javascript
window.showPreview(file, 'gallery');
```

**From Camera:**
```javascript
window.showPreview(file, 'camera');
```

**From Drag-and-Drop:**
```javascript
window.showPreview(file, 'gallery'); // Treated as gallery
```

### 3. Dynamic Button Update
When preview is shown, the button updates automatically:

```javascript
window.updateRetryButton = function(source) {
    if (source === 'camera') {
        // Camera icon and "Retake Photo" text
        btnIcon.className = 'fas fa-camera';
        btnText.textContent = 'Retake Photo';
    } else {
        // Rotate icon and "Choose Another" text
        btnIcon.className = 'fas fa-rotate-right';
        btnText.textContent = 'Choose Another';
    }
};
```

### 4. Context-Aware Click Handler
The button's click action adapts to the source:

```javascript
retryBtn.addEventListener('click', function() {
    hidePreview();

    if (imageSource === 'camera') {
        // Reopen camera for recapture
        window.cameraManager.openCamera();
    } else {
        // Open file gallery for selection
        galleryInput.click();
    }
});
```

## User Experience Flow

### Scenario 1: Camera Capture
```
1. User clicks "Take Photo"
   → Camera opens

2. User captures photo
   → Preview shows with "Retake Photo" button (camera icon)

3. User clicks "Retake Photo"
   → Camera reopens for new capture

4. User captures again
   → New preview shows
```

### Scenario 2: Gallery Selection
```
1. User clicks "Select from Gallery"
   → File browser opens

2. User selects image
   → Preview shows with "Choose Another" button (rotate icon)

3. User clicks "Choose Another"
   → File browser reopens for new selection

4. User selects different image
   → New preview shows
```

### Scenario 3: Mixed Workflow
```
1. User selects from gallery
   → Preview with "Choose Another" button

2. User clicks "Choose Another"
   → File browser opens

3. User selects different image
   → Preview with "Choose Another" button (consistent)

4. User clicks "Cancel"
   → Returns to choice screen

5. User clicks "Take Photo"
   → Camera opens

6. User captures photo
   → Preview with "Retake Photo" button (context switched!)
```

## Visual Changes

### Camera Source Preview
```html
<button type="button" class="btn-custom btn-secondary" id="retryBtn">
    <span class="btn-icon"><i class="fas fa-camera"></i></span>
    <span class="btn-text">Retake Photo</span>
</button>
```

### Gallery Source Preview
```html
<button type="button" class="btn-custom btn-secondary" id="retryBtn">
    <span class="btn-icon"><i class="fas fa-rotate-right"></i></span>
    <span class="btn-text">Choose Another</span>
</button>
```

## Implementation Details

### Modified Files
**`/webapp/static/js/main.js`**

### Key Changes

1. **Added source tracking variable** (line ~19)
   ```javascript
   let imageSource = null;
   ```

2. **Updated showPreview signature** (line ~332)
   ```javascript
   window.showPreview = function(file, source = 'gallery')
   ```

3. **Added source tracking in showPreview** (line ~337)
   ```javascript
   imageSource = source;
   ```

4. **Created updateRetryButton function** (line ~93-107)
   - Updates button icon and text based on source

5. **Updated retry button click handler** (line ~73-90)
   - Checks imageSource and takes appropriate action

6. **Updated all showPreview calls** (lines ~61, ~226, ~1293)
   - Gallery: `showPreview(file, 'gallery')`
   - Camera: `showPreview(file, 'camera')`

## Benefits

### 1. Intuitive User Experience
✅ Button behavior matches user expectations
✅ No confusion about what will happen when clicked
✅ Natural workflow continuity

### 2. Clear Visual Feedback
✅ Button icon indicates the action (camera vs. file selection)
✅ Button text clearly states what will happen
✅ Consistent with the original input method

### 3. Efficient Workflow
✅ Reduces clicks for users wanting to retake photos
✅ No need to go back to main screen
✅ Direct path to desired action

### 4. Professional Polish
✅ Attention to detail in UX design
✅ Smart, adaptive interface
✅ Reduces user frustration

## Debug Console Output

The implementation includes extensive logging for debugging:

```javascript
// When retry button is clicked
"DEBUG: Retry button clicked - source was: camera"
"DEBUG: Reopening camera for recapture"

// or

"DEBUG: Retry button clicked - source was: gallery"
"DEBUG: Opening gallery for new selection"

// When button updates
"DEBUG: Retry button updated for camera recapture"
// or
"DEBUG: Retry button updated for gallery selection"
```

## Edge Cases Handled

### 1. Source Reset on Cancel
When user cancels preview, source is cleared appropriately.

### 2. Mixed Selection Methods
Source correctly updates when switching between camera and gallery.

### 3. Direct File Input
If file input is triggered directly, defaults to 'gallery' behavior.

### 4. Drag-and-Drop
Treated as 'gallery' source since it's a file selection method.

## Testing Checklist

- [✓] Camera capture → "Retake Photo" button shown
- [✓] Gallery selection → "Choose Another" button shown
- [✓] Camera button reopens camera
- [✓] Gallery button reopens file browser
- [✓] Button icon updates correctly
- [✓] Button text updates correctly
- [✓] Source persists across preview
- [✓] Source updates when method changes
- [✓] Works on mobile devices
- [✓] Works on desktop
- [✓] Console logging works for debugging

## Future Enhancements

Potential improvements for this feature:

1. **Animation on button change**
   - Smooth transition when icon/text updates

2. **Tooltip hint**
   - Show tooltip explaining what will happen

3. **Last action memory**
   - Remember user's preferred method across sessions

4. **Quick switch option**
   - Allow user to manually switch between camera/gallery

## Code Maintainability

### Adding New Sources
To add a new image source:

1. Define the source name (e.g., 'clipboard', 'url')
2. Call `showPreview(file, 'yourSource')`
3. Add case in `updateRetryButton()` for custom icon/text
4. Add case in retry button click handler for action

Example:
```javascript
// New clipboard paste source
if (imageSource === 'clipboard') {
    btnIcon.className = 'fas fa-paste';
    btnText.textContent = 'Paste Another';
}
```

### Modifying Button Appearance
Update in `updateRetryButton()` function:
```javascript
window.updateRetryButton = function(source) {
    // Modify icon classes
    // Modify text content
    // Add any additional styling
};
```

## Summary

The context-aware "Choose Another" button provides a **smart, intuitive interface** that:
- 📷 Reopens camera when photo was captured
- 🖼️ Reopens gallery when file was selected
- 🎨 Updates icon and text to match action
- ✨ Creates a seamless, professional user experience

This enhancement eliminates a major UX friction point and demonstrates attention to user workflow details.
