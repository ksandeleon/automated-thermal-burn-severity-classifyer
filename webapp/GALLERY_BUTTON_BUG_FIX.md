# Bug Fix: Gallery Selection Showing "Retake Photo"

## The Problem
When users selected an image from the gallery, the button was showing "Retake Photo" with a camera icon instead of "Choose Another" with a rotate icon.

## Root Cause
The `CameraManager` class was handling **both** gallery and camera file selections, but it was hardcoded to always use `'camera'` as the source, regardless of where the file actually came from.

### Issue Location
**File:** `/webapp/static/js/main.js`

**Problem Code (Line ~1327):**
```javascript
// WRONG: Always used 'camera' source
window.showPreview(file, 'camera');
```

**Problem Code (Lines ~1029-1034):**
```javascript
// WRONG: Didn't pass source parameter
this.galleryInput.addEventListener('change', (e) => this.handleFileSelection(e.target.files));
this.cameraInput.addEventListener('change', (e) => this.handleFileSelection(e.target.files));
```

## The Fix

### 1. Updated Event Listeners (Lines ~1029-1034)
Pass the correct source when files are selected:

```javascript
// FIXED: Pass 'gallery' for gallery input
this.galleryInput.addEventListener('change', (e) =>
    this.handleFileSelection(e.target.files, 'gallery'));

// FIXED: Pass 'camera' for camera input
this.cameraInput.addEventListener('change', (e) =>
    this.handleFileSelection(e.target.files, 'camera'));
```

### 2. Updated handleFileSelection Method (Line ~1305)
Modified to accept and use the source parameter:

```javascript
// FIXED: Accept source parameter with 'gallery' as default
handleFileSelection(files, source = 'gallery') {
    console.log('DEBUG: File selected with source:', source);

    // ...validation...

    // FIXED: Use the passed source instead of hardcoded 'camera'
    window.showPreview(file, source);
}
```

### 3. Updated capturePhoto Method (Line ~1205)
Pass 'camera' source when capturing:

```javascript
// FIXED: Pass 'camera' source for captures
this.handleFileSelection([file], 'camera');
```

## How It Works Now

### Gallery Selection Flow
```
User clicks "Select from Gallery"
    ↓
galleryInput change event fires
    ↓
handleFileSelection([file], 'gallery')  ← 'gallery' source passed
    ↓
window.showPreview(file, 'gallery')  ← Uses 'gallery'
    ↓
imageSource = 'gallery'
    ↓
updateRetryButton('gallery')
    ↓
Button shows: "Choose Another" with rotate icon ✅
```

### Camera Capture Flow
```
User captures photo with camera
    ↓
capturePhoto() creates file from canvas
    ↓
handleFileSelection([file], 'camera')  ← 'camera' source passed
    ↓
window.showPreview(file, 'camera')  ← Uses 'camera'
    ↓
imageSource = 'camera'
    ↓
updateRetryButton('camera')
    ↓
Button shows: "Retake Photo" with camera icon ✅
```

## Files Changed

### `/webapp/static/js/main.js`

**Lines ~1029-1034:** Updated event listener calls
```diff
- this.galleryInput.addEventListener('change', (e) => this.handleFileSelection(e.target.files));
- this.cameraInput.addEventListener('change', (e) => this.handleFileSelection(e.target.files));
+ this.galleryInput.addEventListener('change', (e) => this.handleFileSelection(e.target.files, 'gallery'));
+ this.cameraInput.addEventListener('change', (e) => this.handleFileSelection(e.target.files, 'camera'));
```

**Line ~1305:** Updated method signature
```diff
- handleFileSelection(files) {
+ handleFileSelection(files, source = 'gallery') {
```

**Line ~1320:** Use passed source
```diff
- window.showPreview(file, 'camera');
+ window.showPreview(file, source);
```

**Line ~1205:** Pass source in capturePhoto
```diff
- this.handleFileSelection([file]);
+ this.handleFileSelection([file], 'camera');
```

## Testing

### Test 1: Gallery Selection
1. Click "Select from Gallery"
2. Choose an image
3. **Expected:** Button shows "Choose Another" with 🔄 icon
4. **Console shows:** `source: gallery`

### Test 2: Camera Capture
1. Click "Take Photo"
2. Capture image
3. **Expected:** Button shows "Retake Photo" with 📷 icon
4. **Console shows:** `source: camera`

### Test 3: Button Actions
1. After gallery: Click "Choose Another" → Opens file browser ✅
2. After camera: Click "Retake Photo" → Opens camera ✅

## Debug Console Output

**Gallery Selection:**
```
DEBUG: CameraManager.handleFileSelection called with files: [File] source: gallery
DEBUG: File selected: yourimage.jpg Size: 12345 Source: gallery
DEBUG: showPreview called successfully with source: gallery
DEBUG: updateRetryButton called with source: gallery
DEBUG: Text changed to "Choose Another"
```

**Camera Capture:**
```
DEBUG: CameraManager.handleFileSelection called with files: [File] source: camera
DEBUG: File selected: camera-capture-123456.jpg Size: 54321 Source: camera
DEBUG: showPreview called successfully with source: camera
DEBUG: updateRetryButton called with source: camera
DEBUG: Text changed to "Retake Photo"
```

## Why This Happened

The original implementation had **two separate file handling paths**:
1. Main DOMContentLoaded listener (handled gallery correctly)
2. CameraManager class (handled both but assumed 'camera')

The CameraManager was also listening to the gallery input, and since it processed files second, it overwrote the correct source with 'camera'.

## Prevention

To prevent similar issues:
1. ✅ Always explicitly pass source parameters
2. ✅ Add comprehensive logging to track data flow
3. ✅ Avoid hardcoding values that depend on context
4. ✅ Test both paths (gallery and camera) separately

## Verification

After this fix:
- [✓] Gallery shows "Choose Another"
- [✓] Camera shows "Retake Photo"
- [✓] Button clicks perform correct actions
- [✓] Source tracking works correctly
- [✓] Console logging confirms source
- [✓] No JavaScript errors

## Impact

✅ **User Experience:** Buttons now match user expectations
✅ **Functionality:** Button actions work correctly for each source
✅ **Code Quality:** Proper parameter passing and tracking
✅ **Debugging:** Enhanced logging for troubleshooting

The bug is now completely fixed! 🎉
