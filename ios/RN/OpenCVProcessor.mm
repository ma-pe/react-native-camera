#import "OpenCVProcessor.hpp"
#import <opencv2/opencv.hpp>
#import <opencv2/objdetect.hpp>

@implementation OpenCVProcessor{
    BOOL saveDemoFrame;
    int processedFrames;
    NSInteger expectedFaceOrientation;
    NSInteger objectsToDetect;
}

- (id) init {
    
    saveDemoFrame = true;
    processedFrames = 0;
    expectedFaceOrientation = -1;
    objectsToDetect = -1;
    
    NSString *path = [[NSBundle mainBundle] pathForResource:@"lbpcascade_frontalface_improved.xml"
                                                     ofType:nil];
    
    std::string cascade_path = (char *)[path UTF8String];
    if (!cascade.load(cascade_path)) {
        NSLog(@"Couldn't load haar cascade file.");
    }
    
    if (self = [super init]) {
        // Initialize self
    }
    return self;
}

- (id) initWithDelegate:(id)delegateObj {
    delegate = delegateObj;
    return self;
}

- (void)setExpectedFaceOrientation:(NSInteger)expectedOrientation
{
    expectedFaceOrientation = expectedOrientation;
}

- (void)updateObjectsToDetect:(NSInteger)givenObjectsToDetect
{
    objectsToDetect = givenObjectsToDetect;
}

# pragma mark - OpenCV-Processing

#ifdef __cplusplus

- (void)saveImageToDisk:(Mat&)image;
{
    NSLog(@"----------------SAVE IMAGE-----------------");
    saveDemoFrame = false;
    
    NSData *data = [NSData dataWithBytes:image.data length:image.elemSize()*image.total()];
    CGColorSpaceRef colorSpace;
    
    if (image.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(image.cols,                                 //width
                                        image.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * image.elemSize(),                       //bits per pixel
                                        image.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    UIImageWriteToSavedPhotosAlbum(finalImage, nil, nil, nil);
}

- (int)rotateImage:(Mat&)image;
{
    int orientation = 3;
    //cv::equalizeHist(image, image);
    
    if(expectedFaceOrientation != -1){
        orientation = expectedFaceOrientation;
    } else {
        // rotate image according to device-orientation
        UIDeviceOrientation interfaceOrientation = [[UIDevice currentDevice] orientation];
        if (interfaceOrientation == UIDeviceOrientationPortrait) {
            orientation = 0;
        } else  if (interfaceOrientation == UIDeviceOrientationPortraitUpsideDown) {
            orientation = 2;
        } else  if (interfaceOrientation == UIDeviceOrientationLandscapeLeft) {
            orientation = 1;
        }
    }
    
    switch(orientation){
        case 0:
            transpose(image, image);
            flip(image, image,1);
            break;
        case 1:
            flip(image, image,-1);
            break;
        case 2:
            transpose(image, image);
            flip(image, image,0);
            break;
    }
    
    return orientation;
}

- (float)resizeImage:(Mat&)image width:(float)width;
{
    float scale = width / (float)image.cols;
    
    cv::resize(image, image, cv::Size(0,0), scale, scale, cv::INTER_CUBIC);
    
    return scale;
}

- (void)processImageFaces:(Mat&)image;
{
    int orientation = [self rotateImage:image];
    
    float imageWidth = 480.;
    [self resizeImage:image width:imageWidth];
    float imageHeight = (float)image.rows;
    
    if(saveDemoFrame){
        [self saveImageToDisk:image];
    }
    
    objects.clear();
    cascade.detectMultiScale(image,
                             objects,
                             1.2,
                             3,
                             0,
                             cv::Size(10, 10));
    
    NSMutableArray *faces = [[NSMutableArray alloc] initWithCapacity:objects.size()];
    if(objects.size() > 0){
        for( int i = 0; i < objects.size(); i++ )
        {
            cv::Rect face = objects[i];
            
            NSDictionary *faceDescriptor = @{
                                             @"x" : @((face.x + 0.5 * face.width) / imageWidth),
                                             @"y" : @((face.y + 0.5 * face.height) / imageHeight),
                                             @"width": @(face.width / imageWidth),
                                             @"height": @(face.height / imageHeight),
                                             @"orientation": @(orientation)
                                             };
            
            [faces addObject:faceDescriptor];
        }
    }
    [delegate onFacesDetected:faces];
}

- (BOOL) compareContourAreasReverse: (std::vector<cv::Point>) contour1 contour2:(std::vector<cv::Point>) contour2  {
    double i = fabs( contourArea(cv::Mat(contour1)) );
    double j = fabs( contourArea(cv::Mat(contour2)) );
    return ( i > j );
}

- (void)processImageTextBlocks:(Mat&)image;
{
    [self resizeImage:image width:480.];
    
    int orientation = [self rotateImage:image];
    
    float imageWidth = (float) image.cols;
    float imageHeight = (float) image.rows;
    
    if(saveDemoFrame){
        [self saveImageToDisk:image];
    }
    
    NSMutableArray *detectedObjects = [[NSMutableArray alloc] init];
    
    std::vector<RotatedRect> textBlocks = [self findTextBlocks:image minCharSize:20. removeAtBorder:true];
    
    if(textBlocks.size() > 0){
        for(int i = 0, I = textBlocks.size(); i < I; ++i){
            float aspectRatio = [self aspectRatio:textBlocks[i]];
            
            // if object large enough
            if(aspectRatio >= 5.6 & aspectRatio <= 7.6){
                Point2f rect_points[4];
                textBlocks[i].points( rect_points );
                
                //swap
                if (textBlocks[i].angle < -45.) {
                    float newHeight = textBlocks[i].size.width;
                    textBlocks[i].size.width = textBlocks[i].size.height;
                    textBlocks[i].size.height = newHeight;
                }
                
                float xRel = textBlocks[i].center.x / imageWidth;
                float yRel = textBlocks[i].center.y / imageHeight;
                float widthRel = textBlocks[i].size.width / imageWidth;
                float heightRel = textBlocks[i].size.height / imageHeight;
                //                float sizeRel = widthRel * heightRel;
                
                //            NSLog(@"------");
                //            NSLog(@"--- x: %@", @(xRel));
                //            NSLog(@"--- y: %@", @(yRel));
                //            NSLog(@"--- widthRel: %@", @(widthRel));
                //            NSLog(@"--- heightRel: %@", @(heightRel));
                //            NSLog(@"--- sizeRel: %@", @(sizeRel));
                //            NSLog(@"------");
                
                
                NSDictionary *objectDescriptor = @{
                                                   @"x" : @(xRel),
                                                   @"y" : @(yRel),
                                                   @"width": @(widthRel),
                                                   @"height": @(heightRel),
                                                   @"orientation": @(orientation)
                                                   };
                
                [detectedObjects addObject:objectDescriptor];
            }
        }
    }
    
    [delegate onFacesDetected:detectedObjects];
}

struct line_more {
    bool operator ()(std::pair<Point2f, Point2f> const& a, std::pair<Point2f, Point2f> const& b) const {
        return cv::norm(a.second - a.first) > cv::norm(b.second - b.first);
    }
};

- (float) aspectRatio:(const cv::RotatedRect&)rect {
    
    // Extract the edges of the rotated rect
    Point2f rect_points[4];
    std::vector<std::pair<Point2f, Point2f> > lines;
    rect.points(rect_points);
    
    // Extract the line lengths from the rotated rectangle
    for( int j = 0; j < 4; ++j )
        lines.push_back(std::pair<Point2f, Point2f>(rect_points[j], rect_points[(j+1)%4]));
    
    // Sort line lengths by size
    std::sort(lines.begin(), lines.end(), line_more());
    
    // Return the aspect ratio
    return (cv::norm(lines[0].second - lines[0].first)+cv::norm(lines[1].second - lines[1].first)) / (cv::norm(lines[2].second - lines[2].first)+cv::norm(lines[3].second - lines[3].first));
}

- (void) brightnessContrastAuto:(Mat&)src dst:(Mat&)dst clipHistPercent:(float)clipHistPercent {
    
    CV_Assert(clipHistPercent >= 0);
    CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));
    
    int histSize = 256;
    float alpha, beta;
    double minGray = 0, maxGray = 0;
    
    //to calculate grayscale histogram
    Mat gray;
    if (src.type() == CV_8UC1) gray = src;
    else if (src.type() == CV_8UC3) cvtColor(src, gray, CV_BGR2GRAY);
    else if (src.type() == CV_8UC4) cvtColor(src, gray, CV_BGRA2GRAY);
    if (clipHistPercent == 0)
        minMaxLoc(gray, &minGray, &maxGray); // keep full available range
    else    {
        Mat hist; //the grayscale histogram
        
        float range[] = { 0, 256 };
        const float* histRange = { range };
        bool uniform = true;
        bool accumulate = false;
        calcHist(&gray, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
        
        // calculate cumulative distribution from the histogram
        std::vector<float> accumulator(histSize);
        accumulator[0] = hist.at<float>(0);
        for (int i = 1; i < histSize; ++i)
            accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
        
        
        // locate points that cuts at required value
        float max = accumulator.back();
        clipHistPercent *= (max / 100.0); //make percent as absolute
        clipHistPercent /= 2.0; // left and right wings
        // locate left cut
        minGray = 0;
        while (accumulator[minGray] < clipHistPercent)
            minGray++;
        
        // locate right cut
        maxGray = histSize - 1;
        while (accumulator[maxGray] >= (max - clipHistPercent))
            maxGray--;
    }
    
    // current range
    float inputRange = maxGray - minGray;
    
    alpha = (histSize - 1) / inputRange;   // alpha expands current range to histsize range
    beta = -minGray * alpha;             // beta shifts current range so that minGray will go to 0
    
    // Apply brightness and contrast normalization
    // convertTo operates with saurate_cast
    src.convertTo(dst, -1, alpha, beta);
    
    // restore alpha channel from source
    if (dst.type() == CV_8UC4)  {
        int from_to[] = { 3, 3};
        mixChannels(&src, 4, &dst,1, from_to, 1);
    }
    
    return;
}

- (float) solidityWhite:(const Mat1b&)image rect:(const cv::Rect&)rect threshold:(int)threshold {
    int nWhite = 0;
    
    for (int r = rect.y, R = rect.y+rect.height; r < R; ++r) {
        for (int c = rect.x, C = rect.x+rect.width; c < C; ++c) {
            if (image(r, c) > threshold)
                nWhite++;
        }
    }
    
    return float(nWhite)/rect.area();
}

- (bool) contourTouchesImageBorder:(std::vector<cv::Point>&)contour imageSize:(cv::Size)imageSize {
    cv::Rect bb = boundingRect(contour);
    
    bool retval = false;
    
    int xMin, xMax, yMin, yMax;
    
    xMin = 0;
    yMin = 0;
    xMax = imageSize.width - 1;
    yMax = imageSize.height - 1;
    
    // Use less/greater comparisons to potentially support contours outside of
    // image coordinates, possible future workarounds with cv::copyMakeBorder where
    // contour coordinates may be shifted and just to be safe.
    if( bb.x <= xMin || bb.y <= yMin || bb.width >= xMax || bb.height >= yMax) {
        retval = true;
    }
    
    return retval;
}


- (std::vector<RotatedRect>) findTextBlocks:(Mat&)imageGray  minCharSize:(float)minCharSize removeAtBorder:(bool)removeAtBorder {
    
    // Initialize a elliptic structuring element using a 15x15 grid for the blackhat
    // operation
    Mat blackhatKernel = getStructuringElement(MORPH_ELLIPSE, cv::Size(19.,19.));
    
    // Slightly blur the initial image to reduce noise prior to contrast enhancement
    cv::GaussianBlur(imageGray, imageGray, cv::Size(3, 3), 0);
    
    // Enhance the contrast of the image using CLAHE
    Mat imageGrayClahe;
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(1);
    clahe->apply(imageGray,imageGrayClahe);
    
    // Blur the image again to remove some noise created by the contrast enhancement
    cv::GaussianBlur(imageGrayClahe, imageGray, cv::Size(1, 1), 0);
    
    // Apply a blackhat operation to extract black elements on brighter background
    morphologyEx(imageGray, imageGray, MORPH_BLACKHAT, blackhatKernel);
    
    // Normalize the image histogram to acquire a baseline
    [self brightnessContrastAuto:imageGray dst:imageGray clipHistPercent:0.];
    
    // Remove weak regions as they most likely did not originate
    bitwise_not(imageGray,imageGray);
    clahe->apply(imageGray, imageGrayClahe);
    
    // Perform adaptive thresholding to extract text areas w.r.t. background
    adaptiveThreshold(imageGrayClahe, imageGray, 255,ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 55, 35);
    
    // Dilate and erode to get rid of small speckles
    dilate(imageGray, imageGray, Mat(), cv::Point(-1, -1), 1, 1, 1);
    erode(imageGray, imageGray, Mat(), cv::Point(-1, -1), 1, 1, 1);
    
    // Determine contours from image
    std::vector< std::vector<cv::Point> > contours;
    std::vector<Vec4i> hierarchy;
    findContours(imageGray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    // Prepare storage for removal of spurious contours and final candidates
    std::vector< std::vector<cv::Point> > contoursDrop;
    std::vector<float> candidateSizes;
    
    // Veto speckles and small non-continuous areas based on size and other image
    // moments / parameters
    for (int i = 0, I = contours.size(); i < I; ++i) {
        
        // Veto contours which are too small to be actual useful characters
        float area = contourArea(contours[i]);
        if (area < minCharSize) {
            contoursDrop.push_back(contours[i]);
            continue;
        }
        
        // Veto contours with largely deviating contour vs. bounding rectangle areas
        cv::Rect brect = boundingRect( Mat(contours[i]) );
        //NSLog(@"%d", brect.x);
        float brectArea = (float)brect.width * (float)brect.height;
        if ( area / brectArea < 0.2 || area / brectArea > 5.0 ) {
            contoursDrop.push_back(contours[i]);
            continue;
        }
        
        // Veto contours with aspect ratios largely deviating from that of characters
        float aspect = (float)brect.width/(float)brect.height;
        if (aspect < 0.25 || aspect > 4.0) {
            contoursDrop.push_back(contours[i]);
            continue;
        }
        
        // Veto contours with low white pixel solidity
        if ([self solidityWhite:imageGray rect:brect threshold:200] < 0.3) {
            contoursDrop.push_back(contours[i]);
            continue;
        }
        
        // Store the bounding rectangle area in the candidate vector
        candidateSizes.push_back(brectArea);
    }
    
    // Safety Guard
    if(candidateSizes.size() < 10){
        return std::vector<RotatedRect>();
    }
    
    // Drop all contours vetoed in the loop above by setting their pixels to zero
    drawContours( imageGray, contoursDrop, -1, Scalar(0), CV_FILLED, 8);
    
    // Create 1D matrix from candidate contours =
    Mat dataMat(candidateSizes.size(), 1, CV_32FC1, &candidateSizes[0]);
    
    // Perform k-means clustering to extract centers of contour size distribution
    // TODO: This should be an n-means algorithm, really, as we don't know k by definition
    std::vector<int> labels;
    std::vector<float> centers;
    kmeans(dataMat, 1, labels, TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 10, 0.1), 3, KMEANS_PP_CENTERS, centers);
    
    // If no cluster was found, return empty result vector
    if (centers.size() == 0 || labels.size() == 0)
        return std::vector<RotatedRect>();
    
    // Apply an adaptive closing operation using an elliptic structuring element with
    // a size proportional to the square root of the main cluster's average area
    Mat closeKernel = getStructuringElement(MORPH_ELLIPSE, cv::Size(1.2*sqrt(centers[0]),1.2*sqrt(centers[0])));
    morphologyEx(imageGray, imageGray, MORPH_CLOSE, closeKernel, cv::Point(-1,-1), 1);
    
    // Find a new set of contours (this time at least the individual lines should
    // be merged / closed)
    contoursDrop.clear();
    findContours(imageGray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (int i = 0, I = contours.size(); i < I; ++i) {
        //contours[i].
        // Veto all contours which are not at least 15x the size of the main cluster's average area
        // Since we expect only few,
        if (contourArea(contours[i]) < centers[0]*15.)
            contoursDrop.push_back(contours[i]);
    }
    
    // Drop all contours vetoed in the loop above by setting their pixels to zero
    drawContours( imageGray, contoursDrop, -1, Scalar(0), CV_FILLED, 8);
    
    // Perform one last closing operation using the same kernel as above to merge
    // the remaining contours (most likely only the individual lines)
    morphologyEx(imageGray, imageGray, MORPH_CLOSE, closeKernel, cv::Point(-1,-1), 3);
    
    // Extract the final contour(s) and sort by area
    findContours(imageGray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //std::sort(contours.begin(), contours.end(), compareContourAreasReverse);
    
    // Veto all contours that touch the border (if requested)
    std::vector<RotatedRect> minRects;
    for (int i = 0, I = contours.size(); i < I; ++i) {
        for (int h = 0, H = contours[i].size(); h < H; ++h) {
            if(contours[i].at(h).y < 0){
                NSLog(@"not----------------- %d", contours[i].at(h).y);
            }
        }
        // Remove contours that touch the border, as it is highly unlikely that they actually
        // represent any reasonable text
        if (removeAtBorder && [self contourTouchesImageBorder:contours[i] imageSize:imageGray.size()])
            continue;
        
        // Store the candidate
        minRects.push_back(minAreaRect(Mat(contours[i])));
    }
    
    // Return resulting vector
    return minRects;
}



- (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection
{
    // https://github.com/opencv/opencv/blob/master/modules/videoio/src/cap_ios_video_camera.mm
    if(processedFrames % (objectsToDetect == 1 ? 5 : 10) == 0){
        (void)captureOutput;
        (void)connection;
        
        // convert from Core Media to Core Video
        CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
        CVPixelBufferLockBaseAddress(imageBuffer, 0);
        
        void* bufferAddress;
        size_t width;
        size_t height;
        size_t bytesPerRow;
        
        int format_opencv = CV_8UC1;
        
        bufferAddress = CVPixelBufferGetBaseAddressOfPlane(imageBuffer, 0);
        width = CVPixelBufferGetWidthOfPlane(imageBuffer, 0);
        height = CVPixelBufferGetHeightOfPlane(imageBuffer, 0);
        bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(imageBuffer, 0);
        
        // delegate image processing to the delegate
        cv::Mat image((int)height, (int)width, format_opencv, bufferAddress, bytesPerRow);
        
        switch(objectsToDetect){
            case 0:
                [self processImageFaces:image];
                break;
            case 1:
                [self processImageTextBlocks:image];
                break;
        }
        
        // cleanup
        CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
    }
    processedFrames++;
}
#endif

@end

