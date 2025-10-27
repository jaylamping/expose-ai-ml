# Debugging Guide for Bot Detection API

## Overview

This guide explains the verbose logging that has been added to help debug the 500 error in the Bot Detection API.

## Logging Configuration

### 1. Logging Setup

- **Log Level**: DEBUG (most verbose)
- **Output**: Both console and file (`bot_detection.log`)
- **Format**: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

### 2. Files Modified for Logging

#### `api/expose.py`

- Added comprehensive logging to the main API endpoint
- Added detailed error handling with stack traces
- Added logging to model initialization process
- Added logging to each analysis stage (fast, deep, statistical)
- Fixed parent context handling issue

#### `models/detectors/fast_detector.py`

- Added logging to `analyze_user_comments` method
- Added logging to `predict_batch` method
- Added detailed error handling with stack traces

## Key Logging Points

### 1. Request Processing

```
INFO - Starting analysis for user: {user_id}
DEBUG - Request details - Comments: {count}, Options: {options}
DEBUG - Validating request...
DEBUG - Request validation passed - {count} comments to analyze
```

### 2. Model Initialization

```
INFO - Starting model initialization process...
DEBUG - Initializing preprocessor...
INFO - Preprocessor initialized successfully
DEBUG - Initializing fast detector...
INFO - Fast detector initialized successfully
```

### 3. Analysis Stages

```
INFO - Stage 1: Fast screening for user {user_id}
DEBUG - FastDetector: Starting analysis of {count} comments
DEBUG - FastDetector: Using pipeline for batch processing...
INFO - Fast screening completed - Bot score: {score}, Confidence: {confidence}
```

### 4. Error Handling

```
ERROR - Unexpected error in analyze_user: {error}
ERROR - Full traceback: {traceback}
ERROR - FastDetector: Error in predict_batch: {error}
ERROR - FastDetector: predict_batch traceback: {traceback}
```

## Debugging Scripts

### 1. `run_debug.py`

- Runs the API server in debug mode
- Sets up proper environment variables
- Provides clear startup messages

### 2. `test_debug.py`

- Tests the API after it's running
- Waits for server to be ready
- Sends test requests and logs responses
- Helps identify where the 500 error occurs

### 3. `debug_api.py`

- Simple test script for manual testing
- Can be run independently

## How to Use

### 1. Start the Server

```bash
python run_debug.py
```

### 2. In Another Terminal, Test the API

```bash
python test_debug.py
```

### 3. Check Logs

- Console output shows real-time logs
- `bot_detection.log` file contains all logs
- Look for ERROR messages and stack traces

## Common Issues to Look For

### 1. Model Loading Issues

- Look for errors during model initialization
- Check if all required models are downloaded
- Verify device availability (CPU/GPU)

### 2. Request Format Issues

- Check if the request matches the expected schema
- Look for validation errors
- Verify comment structure

### 3. Memory Issues

- Look for CUDA out of memory errors
- Check if models are too large for available memory
- Verify batch size settings

### 4. Dependencies Issues

- Check if all required packages are installed
- Look for import errors
- Verify Python version compatibility

## Log Analysis Tips

1. **Start from the top**: Look for the first ERROR message
2. **Follow the stack trace**: It will show exactly where the error occurred
3. **Check model initialization**: Most 500 errors happen during model loading
4. **Look for timeout errors**: Model loading can take time
5. **Check memory usage**: Large models might cause memory issues

## Expected Log Flow

1. Server startup and model initialization
2. Health check endpoint test
3. Request validation
4. Model initialization (if not already done)
5. Fast analysis stage
6. Deep analysis stage (if not skipped)
7. Statistical analysis stage
8. Ensemble scoring
9. Response generation

If any step fails, the logs will show exactly where and why.
