# Testing Guide - Context-Aware Retry Button

## How to Test

### Test 1: Gallery Selection
1. Open the webapp
2. Open browser DevTools (F12)
3. Go to Console tab
4. Click "Select from Gallery"
5. Choose an image file
6. **Expected Console Output:**
   ```
   DEBUG: ============================================
   DEBUG: showPreview called
   DEBUG: File: [your-filename.jpg]
   DEBUG: Source: gallery
   DEBUG: ============================================
   DEBUG: imageSource variable set to: gallery
   ...
   DEBUG: updateRetryButton called with source: gallery
   DEBUG: Icon changed to rotate-right
   DEBUG: Text changed to "Choose Another"
   ```
7. **Expected Button:** "Choose Another" with rotate icon 🔄

### Test 2: Camera Capture
1. Open the webapp (refresh to clear state)
2. Keep DevTools Console open
3. Click "Take Photo"
4. Allow camera permissions
5. Capture a photo
6. **Expected Console Output:**
   ```
   DEBUG: ============================================
   DEBUG: showPreview called
   DEBUG: File: camera-capture-[timestamp].jpg
   DEBUG: Source: camera
   DEBUG: ============================================
   DEBUG: imageSource variable set to: camera
   ...
   DEBUG: updateRetryButton called with source: camera
   DEBUG: Icon changed to camera
   DEBUG: Text changed to "Retake Photo"
   ```
7. **Expected Button:** "Retake Photo" with camera icon 📷

### Test 3: Button Click Behavior

#### After Gallery Selection:
1. Click "Choose Another" button
2. **Expected Console Output:**
   ```
   DEBUG: Retry button clicked - source was: gallery
   DEBUG: Opening gallery for new selection
   ```
3. **Expected Behavior:** File browser opens

#### After Camera Capture:
1. Click "Retake Photo" button
2. **Expected Console Output:**
   ```
   DEBUG: Retry button clicked - source was: camera
   DEBUG: Reopening camera for recapture
   ```
3. **Expected Behavior:** Camera reopens

## Troubleshooting

### Issue: Button shows wrong text/icon

**Check Console For:**
1. Is `showPreview` being called with correct source?
   - Look for: `DEBUG: Source: gallery` or `DEBUG: Source: camera`

2. Is `updateRetryButton` being called?
   - Look for: `DEBUG: updateRetryButton called with source:`

3. Are button elements found?
   - Look for: `DEBUG: Button elements found: { btnIcon: true, btnText: true }`

**If source is wrong:**
- Check the calling code is passing the correct source parameter
- Gallery calls should use: `window.showPreview(file, 'gallery')`
- Camera calls should use: `window.showPreview(file, 'camera')`

**If button elements not found:**
- Button might not be rendered yet
- Check if `retryBtn` variable is defined
- Try adding delay before updating

### Issue: Button doesn't update

**Possible causes:**
1. JavaScript error preventing execution
2. Button not found in DOM
3. Source not being tracked
4. Function not being called

**Check:**
- Any errors in console?
- Does `updateRetryButton` function exist?
- Is `imageSource` variable accessible?

### Issue: Button clicks do wrong action

**Check:**
1. Console output when clicking button
2. Value of `imageSource` when button clicked
3. Correct branch of if/else being executed

## Expected Behavior Summary

| User Action | Source Value | Button Icon | Button Text | Button Clicks → |
|-------------|-------------|-------------|-------------|----------------|
| Select from Gallery | `'gallery'` | `fa-rotate-right` | "Choose Another" | Opens file browser |
| Take Photo | `'camera'` | `fa-camera` | "Retake Photo" | Reopens camera |
| Drag & Drop | `'gallery'` | `fa-rotate-right` | "Choose Another" | Opens file browser |

## Debug Checklist

When testing, verify each step:

- [ ] Gallery selection logs `Source: gallery`
- [ ] Camera capture logs `Source: camera`
- [ ] `imageSource` variable is set correctly
- [ ] `updateRetryButton` is called
- [ ] Button icon updates to correct class
- [ ] Button text updates to correct string
- [ ] Button click logs correct source
- [ ] Button click performs correct action
- [ ] No JavaScript errors in console

## Known Working States

✅ **Gallery → Choose Another → Gallery**
```
Select Gallery → Shows "Choose Another" → Click → Opens Gallery
```

✅ **Camera → Retake Photo → Camera**
```
Take Photo → Shows "Retake Photo" → Click → Opens Camera
```

✅ **Gallery → Cancel → Camera → Retake Photo**
```
Select Gallery → Cancel → Take Photo → Shows "Retake Photo"
```

## Browser Console Commands

You can manually check state in console:

```javascript
// Check current image source
console.log('Current source:', imageSource);

// Manually update button
window.updateRetryButton('gallery');
window.updateRetryButton('camera');

// Check if button exists
console.log('Button exists:', !!document.getElementById('retryBtn'));

// Get button current text
console.log('Button text:', document.getElementById('retryBtn')?.querySelector('.btn-text')?.textContent);
```

## What to Report

If you find an issue, please provide:
1. Steps to reproduce
2. Full console output (especially the DEBUG lines)
3. Screenshot of the button
4. Expected vs actual behavior

The enhanced logging will help identify exactly where things go wrong!
