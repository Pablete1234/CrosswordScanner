
disp("Baseline acc:" + findDatasetAcc("Crossword"));
disp("Rotaated acc:" + findDatasetAcc("rotated"));
disp("Noised acc:" + findDatasetAcc("noised"));
% Mixed dataset contains images with blur. Those images find no lines and
% just straight fail.
disp("Mixed acc:" + findDatasetAcc("mixed"));

%acc = treatImageAndFindResult(imread("dataset/blured/blured_2.png", "BackgroundColor", [1,1,1]));

function accuracy = findDatasetAcc(type)
    accuracy = 0;
    for cw = 1:100
        try 
            im = imread("dataset/" + type + "/" + type + "_" + cw + ".png",  "BackgroundColor", [1,1,1]);
            expected = csvread("dataset/Crossword/Crossword_" + cw + ".csv");
            accuracy = accuracy + findImageAcc(im, expected);
        catch
            disp("Error processing: " + type + "_" + cw);
        end
    end
    
    accuracy = accuracy / 100;
end


function accuracy = findImageAcc(im, expected)
    result = treatImageAndFindResult(im);

    [m,n] = size(expected);
    accuracy = sum(result == expected,'all') / (m * n);
end


function result = treatImageAndFindResult(im)
    im_gray = rgb2gray(im);
    im = findAndCorrect(im_gray);

    points = findLines(im);
    points = expandAndMergeLines(im, points, 20);

    [hor , ver] = categorizeLines(points);

    result = getBoardResult(im, hor, ver);
    
    %im = insertShape(im,'Line', hor, 'LineWidth',5,'Color','green');
    %im = insertShape(im,'Line', ver, 'LineWidth',5,'Color','green');
    %figure, imshow(im);
end



function corrected = findAndCorrect(im)
    % Binarize & invert image
    im_bin = ~imbinarize(im);

    % Dilate lines to ensure they connect
    im_mask = imdilate(im_bin, strel('disk', 2));
    % Find largest area (crossword)
    im_mask = bwareafilt(im_mask , 1,'largest');
    % Close in all the squares in the crossword
    im_mask = imclose(im_mask , strel('disk', 50));

    % Find lines around the square
    points = findLines(im_mask);
    points = expandAndMergeLines(im, points, 50);
    
    %im = insertShape(im,'Line', points, 'LineWidth',5,'Color','green');

    % Locate the corners of lines intersecting
    corners = findCorners(points);

    % Apply image correction to center & remove rotation
    corrected = imcorrect((im.*uint8(im_mask))+uint8(~im_mask)*255, corners);
end

function result = findLines(im_gray)   
    BW = edge(im_gray,'canny', 0.1);
    BW = imdilate(BW, [strel('line', 3, 0) strel('line',3,90)]);
    BW = imclose(BW, strel('line', 5, 0));
    BW = imclose(BW, strel('line', 5, 90));
    %BW = imdilate(BW, strel('line', 4, 90));

    [H,T,R] = hough(BW, 'RhoResolution', 1);
    P  = houghpeaks(H,100);
    lines = houghlines(BW,T,R,P, 'FillGap', 8, 'MinLength', 20);

    points1 = reshape([lines.point1],2,[]).';
    points2 = reshape([lines.point2],2,[]).';

    result = [points1,points2];
    
    %BW = insertShape(BW * 255,'Line',result,'LineWidth',3,'Color','green');
    %figure, imshow(BW);
end

function result = expandAndMergeLines(im, points, mergeDist)
    lengths = zeros(size(points, 1), 1);

    k = 1;
    while k <= size(points, 1)
        if (points(k,3) < points(k, 1))
            points(k,:) = [points(k,3), points(k,4), points(k,1), points(k,2)];
        end

        diffx = points(k, 3) - points(k, 1);
        diffy = points(k, 4) - points(k, 2);

        if (diffx == 0)
            diffx = 0.00001;
        end

        lengths(k) = sqrt(diffx ^ 2 + diffy ^ 2);
        line = [diffy / diffx, -1, points(k,2) - (points(k,1) * diffy / diffx)];
        points(k, :) = lineToBorderPoints(line, size(im));

        for orig = 1:(k-1)
            dist = sqrt(sum((points(k,1:2) - points(orig,1:2)) .^ 2));
            dist = dist * sqrt(sum((points(k,3:4) - points(orig,3:4)) .^ 2));

            if (dist < (mergeDist ^ 2))
                points(k,:) = (points(orig,:) * lengths(orig)^10 +...
                    points(k,:) * lengths(k)^10) / (lengths(orig)^10 + lengths(k)^10);
                lengths(k) = lengths(orig) + lengths(k);

                points(orig,:) = [];
                lengths(orig) = [];

                k = k - 1;
                break;
            end
        end
        k = k + 1;
    end
    result = points;
end

function result = findCorners(points)
    corners = [
        Inf, Inf; % min x, min y
        -Inf, Inf; % max x, min y
        Inf, -Inf; % min x, max y
        -Inf, -Inf % max x, max y
    ];

    for k = 1:size(points,1)
        for l = k+1:size(points,1)
            dot = intersect(points(k,:), points(l,:));
            if size(dot) > 0
                %im = insertShape(im,'Circle',[dot(1), dot(2), 10],'LineWidth',3,'Color','blue');

                if (dot(1) + dot(2) < corners(1, 1) + corners(1,2))
                    corners(1,:) = dot(:);
                end
                if (-dot(1) + dot(2) < -corners(2, 1) + corners(2,2))
                    corners(2,:) = dot(:);
                end
                if (dot(1) + -dot(2) < corners(3, 1) + -corners(3,2))
                    corners(3,:) = dot(:);
                end
                if (-dot(1) + -dot(2) < -corners(4, 1) + -corners(4,2))
                    corners(4,:) = dot(:);
                end
            end
        end
    end
    result = corners;
end

function result = imcorrect(im, corners)
    pad = 10;
    res = 2;

    maxX = (max(corners(:,1)) - min(corners(:,1))) * res;
    maxY = (max(corners(:,2)) - min(corners(:,2))) * res;
    
    x1 = corners(:,1);
    y1 = corners(:,2);
    x2 = [0; maxX; 0; maxX];
    y2 = [0; 0; maxY; maxY];

    M = [];
    for i=1:length(x1)
        M = [ M ;
            x1(i) y1(i) 1 0 0 0 -x2(i)*x1(i) -x2(i)*y1(i) -x2(i);
            0 0 0 x1(i) y1(i) 1 -y2(i)*x1(i) -y2(i)*y1(i) -y2(i)];
    end
    [u,s,v] = svd( M );
    H = reshape( v(:,end), 3, 3 )';
    H = H / H(3,3);

    transf = projective2d(H');
    panorama = imref2d([ceil(maxX) + pad * 2 ,ceil(maxY) + pad * 2],... 
            [-pad, ceil(maxX) + pad], [-pad, ceil(maxY) + pad]);
    result = imwarp(im, transf, 'OutputView', panorama);
end

function result = intersect(line1, line2)
    x1 = line1(1);
    y1 = line1(2);
    x2 = line1(3);
    y2 = line1(4);
    
    x3 = line2(1);
    y3 = line2(2);
    x4 = line2(3);
    y4 = line2(4);
    
    m1 = (y2 - y1) / (x2 - x1);
    m2 = (y4 - y3) / (x4 - x3);
    
    angle = atan(abs((m2 - m1) / (1 + m1 * m2)));
    
    if abs(angle - (pi / 2)) > 0.7
        result = [];
    else
        b1 = y1 - (x1 * m1);
        b2 = y3 - (x3 * m2);
        
        x = (b2 - b1) / (m1 - m2);
        y = m1 * x + b1;

        result = [x, y];
    end
end

function [hor, ver] = categorizeLines(points)
    hor = [];
    ver = [];
    for k=1:length(points)
        p = points(k,:);
        
        diffx = abs(p(1) - p(3));
        diffy = abs(p(2) - p(4));
        
        if (diffx * 0.05 > diffy) % Clearly horizontal
            hor = [hor; p];
        elseif (diffy * 0.05 > diffx) % Clearly vertical
            ver = [ver; p];
        end
        % other (diagonal) lines are simply ignored
    end
    
    %im = insertShape(im,'Line', hor, 'LineWidth',5,'Color','green');
    %im = insertShape(im,'Line', ver, 'LineWidth',5,'Color','green');
    %figure, imshow(im);

    hor = cleanupLines(hor, 2, 4);
    ver = cleanupLines(ver, 1, 3);
    
    %im = insertShape(im,'Line', hor, 'LineWidth',5,'Color','green');
    %im = insertShape(im,'Line', ver, 'LineWidth',5,'Color','green');
    %figure, imshow(im);
end

function lines = cleanupLines(lines, idx1, idx2)
    lines = sortrows(lines);
    linecount = size(lines,1);
    
    % Find diffs between lines in the corresponding axis
    diffs = zeros(linecount-1, 1);
    last = (lines(1,idx1) + lines(1, idx2)) / 2;
    for k = 2:size(lines,1)
        curr = (lines(k,idx1) + lines(k, idx2)) / 2;
        diffs(k-1) = curr - last;
        last = curr;
    end
    % Find median value +-50% margin
    med = median(diffs);
    minDiff = med * 0.5;
    maxDiff = med * 1.5;
    
    % Add missing lines or remove extra lines
    last = (lines(1,idx1) + lines(1, idx2)) / 2;
    k = 2;
    while k <= linecount
        curr = (lines(k,idx1) + lines(k, idx2)) / 2;
        diff = curr - last;
        if (diff < minDiff)
            lines(k,:) = []; % Remove extra line
            linecount = linecount - 1;
            k = k - 1;
        elseif (diff > maxDiff)
            missing = round(diff / med) - 1;
            
            base = lines(k-1,:);
            lineDist = lines(k,:) - base;
             % Add missing line
            for j = 1:missing
                lines = [lines; base + lineDist * j / (missing + 1)];
            end
            
            last = curr;
        else
            last = curr;
        end
        k = k + 1;
    end
    
    lines = sortrows(lines);
end


function result = getBoardResult(im, hor, ver)

    im_gray = im2gray(im);
    bin = imbinarize(im_gray);

    % Find regions for better-matched OCR than vertical lines
    stats = getRegions(hor, ver, bin);
    centroids = reshape([stats.Centroid],2,[]).';

    result = zeros(length(ver)-1, length(hor)-1);

    for k=1:length(hor)-1
        linehor1 = hor(k,:);
        linehor2 = hor(k+1,:);

        for j=1:length(ver)-1
            linever1 = ver(j,:);
            linever2 = ver(j+1,:);

            % Find corners for this cell
            corner1 = ceil(intersect(linehor1,linever1));
            corner2 = ceil(intersect(linehor2,linever2));

            % Find cell & report result
            num = getCellResult(corner1, corner2, centroids, stats, bin, im_gray);
            
            result(k, j) = num;
        end
    end
end

function stats = getRegions(hor, ver, bin_im)
    % find first & last hor & ver lines
    linehor1 = hor(1,:);
    linehor2 = hor(end,:);
    linever1 = ver(1,:);
    linever2 = ver(end,:);

    % find top-left & bottom-right corners
    corner1 = ceil(intersect(linehor1,linever1));
    corner2 = ceil(intersect(linehor2,linever2));

    % Estimate an avg size per cell
    cells = (length(ver)-1) * (length(hor)-1);
    board_size = corner2-corner1;
    area = board_size(1)*board_size(2)/cells; 

    % Find all cell regions in binary image
    stats = regionprops(bin_im);
    % Filter areas with +-20% width & +- 20% height diff from avg expected
    min_area = area*0.8^2;
    max_area = area*1.2^2;
    stats = stats([stats.Area] < max_area);
    stats = stats([stats.Area] > min_area);
end

function num = getCellResult(corner1, corner2, centroids, stats, bin, im_gray)
    % Get center
    centroid = (corner2+corner1)/2;
    % Get padded-in 20% points
    minpoint = floor((corner1*4+corner2)/5);
    maxpoint = ceil((corner1+corner2*4)/5);

    % If fully true or false, detect as white or black square
    u = unique(bin(minpoint(2):maxpoint(2), minpoint(1):maxpoint(1)));
    if length(u) == 1
        if u(1) 
            num = 0;
        else
            num = -1;
        end
        return;
    end
    
    % If mixed, perform OCR to find number

    % Find nearest center region
    nearest = stats(dsearchn(centroids,centroid)).BoundingBox;
    nearest_center = nearest([1,2])+ nearest([3,4])/2;

    if any(nearest_center < corner1) || any(nearest_center > corner2)
        % No proper bounding box region exists for this square, use our own
        nearest = [corner1 corner2-corner1];
    end
    
    % Cut 5% from top-left & 15% from bottom to remove any cornering lines
    nearest = [nearest(1)+nearest(3)*0.05 nearest(2)+nearest(4)*0.05 nearest(3)*0.8 nearest(4)*0.8];
    nearest = round(nearest); % ensure int
    
    % Create cut image with just the specific cell to search
    new_im = im_gray(nearest(2):nearest(2)+nearest(4), nearest(1):nearest(1)+nearest(3));
    %figure, imshow(new_im);
    num = ocr(new_im,'TextLayout','Block','CharacterSet','0123456789').Text;
    num(num == ' ') = []; % Remove whitespaces
    num = str2num(num);
    if isempty(num)
        num = -2; % if no number was found, set -2 as result
    end    
end