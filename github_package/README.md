# Raptur - Audio Mixing App

This folder is ready to be uploaded to GitHub to build a macOS desktop application.

## How to Build (One-Time Setup)

### Step 1: Create a GitHub Account
1. Go to https://github.com
2. Click "Sign up" and follow the steps
3. Verify your email address

### Step 2: Create a New Repository
1. Click the **+** button in the top right corner
2. Select "New repository"
3. Name it: `raptur-app`
4. Make sure "Public" is selected
5. Click "Create repository"

### Step 3: Upload These Files
1. On your new repository page, click "uploading an existing file"
2. Drag ALL files from this folder (including the `.github` folder) into the upload area
3. Click "Commit changes"

### Step 4: Run the Build
1. Click the "Actions" tab at the top
2. Click "Build Raptur for macOS" in the left sidebar
3. Click "Run workflow" button (dropdown on the right)
4. Click "Run workflow" again to confirm
5. Wait 5-10 minutes for the build to complete (you'll see a green checkmark)

### Step 5: Download Your App
1. Click on the completed workflow run
2. Scroll down to "Artifacts"
3. Click "Raptur-macOS" to download the zip file

### Step 6: Upload to Replit
1. Rename the file to `Raptur.zip`
2. Drag it into your Replit project files
3. The download button in your app will now work!

## Files Included
- `app.py` - Main Raptur application
- `run_raptur.py` - Desktop launcher script
- `requirements.txt` - Python dependencies
- `attached_assets/` - Logo and images
- `.github/workflows/build-macos.yml` - Build automation
