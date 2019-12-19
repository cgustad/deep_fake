## Libraries
library(av)
library(tidyverse)
library(stringr)
library(opencv)
library(magick)
library(ggplot2)
library(scales)
library(imager)
library(reshape2)
library(ggplot2)
library(rjson)
library(caret)
library(keras)


root_path <- '~/github/kaggle/deepfake/'
setwd(root_path)

test_video_path <- "./input/test_videos/"
train_video_path <- "./input/train_sample_videos/"
root_test_path <- "./dat_test/"
root_train_path <- "./dat_train/"
dir.create(root_train_path,showWarnings = FALSE)
dir.create(root_test_path,showWarnings = FALSE)
#train_image_path <- "./dat/"

test_video_name <- "aagfhgtpmv.mp4"
test_video_path <- paste0(train_video_path,test_video_name)


analyze_image <- function(img_dir,mask_dir){
    ## Run img
    img <- load.image(img_dir)
    ## Total red
    total_red <- sum(img[,,,1])
    ## Total green
    total_green <- sum(img[,,,2])
    ## Total blue
    total_blue <- sum(img[,,,3])
    ## Total color
    tot_col <- total_red + total_green + total_blue  
    ## Run mask
    mask <- load.image(mask_dir)
    ## Size face
    size_face <- sum(mask[,,1,]!= 0) 
    ## Return vector
    out_vec <- tibble(red=total_red,green=total_green,blue=total_blue,tot_color=tot_col,size=size_face)
    return(out_vec)
}
make_analysis_video <- function(dataframe,vid_name,size_plot = FALSE,col_plot = TRUE,makevids = TRUE){
    # Create directory for plots
    dir.create(paste0('./dat/img/',vid_name,'/plot/'),showWarnings = FALSE)
    dir.create(paste0('./dat/vids/',vid_name,'/'),showWarnings = FALSE)
    ## Cplot RGB values
    if(col_plot == TRUE){
        make_rgb <- function(save = FALSE){
            ## Create dir
            dir.create(paste0('./dat/img/',vid_name,'/plot/RGB/'),showWarnings = FALSE)
            rgb_paths = list()
            for(curr_frame in 2:nrow(dataframe)){
                ## Find maximal and minimal y value for nicer look
                min_y <- dataframe %>% select(red,blue,green) %>% min
                max_y <- dataframe %>% select(red,blue,green) %>% max
                ##Plot
                out_frame <- dataframe %>%
                    filter(frame <= curr_frame) %>% 
                    select(red,blue,green,frame) %>%
                    ggplot(.,aes(x=frame)) +
                    geom_line(aes(y=red),color="red") +
                    geom_line(aes(y=blue),color="blue") +
                    geom_line(aes(y=green),color="green") +
                    ylim(min_y,max_y) +
                    xlim(curr_frame - 10, curr_frame + 10)
                
                ## Save image
                if(save==TRUE){
                    plot_path <- paste0('./dat/img/',vid_name,'/plot/RGB/RGB_plot',curr_frame,'.jpg')
                    ggsave(plot_path)
                    rgb_paths[curr_frame] <<- plot_path
                }
                print(out_frame)
            }
        }
        ## Generate video of plots
        if(makevids== TRUE){
            vid_path <- paste0('./dat/vids/',vid_name,'/RGB.mp4')
            print(paste0('Generating video: ',vid_path))
            av_capture_graphics(make_rgb(), vid_path, 1280, 720, res = 144, framerate = 30)
        }
        else if (makevids == FALSE){
            make_rgb(save==TRUE)
        }
    }

    ## Size of head plot   
    if(size_plot == TRUE){
        for(curr_frame in 1:nrow(dataframe)){
    
            print('generating size plot')
            current_df <- dataframe %>%
                filter(frame <= curr_frame) %>%
                ggplot(.,aes(x=frame, y=size)) + geom_line()
            ggsave(paste0('./dat/img/',vid_name,'/plot/size/',curr_frame,'.jpg'))
        }
    }
    
}


run_video <- function(video_path,make_img = FALSE,analyze_img = FALSE,create_vid = FALSE,type='train'){
    if(type=='train')
    {
        root_dat_path <- root_train_path
    }
    else if(type=='test')
    {
        root_dat_path <- root_test_path
    }
    # Video name
    vid_name <- str_split(tail(str_split(video_path,'/')[[1]],1),'\\.')[[1]][1]
    video_data_dir <- paste0(root_dat_path,vid_name,'/')
    dir.create(video_data_dir,showWarnings = FALSE)
    print(paste0('Writing data at:', video_data_dir))
    # Generate images from video
    if(make_img == TRUE){
        # Set path and create directory
        destdir <- paste0(video_data_dir,'frames')
        dir.create(destdir,showWarnings = FALSE)
        ## Grab images
        image_paths <- av_video_images(video_path,destdir=destdir,format='jpg')
    }
    # Analyze output from videos
    if(analyze_img == TRUE){
        dir.create(paste0(video_data_dir,'mask/'),showWarnings = FALSE)
        # Initalize variables
        img_dat <- data.frame(red=numeric(),green=numeric(),blue=numeric(),tot_color=numeric(),size=integer()) %>% as_tibble()
        for(image in image_paths){
            image_name <- tail(str_split(image,'/')[[1]],1)
            ## Generate face mask data
            ocv_img <- ocv_read(image)
            face_mask_img <- ocv_facemask(ocv_img)
            destdir <- paste0(video_data_dir,'mask/',image_name)
            mask_path <- ocv_write(face_mask_img,destdir)
            
            ## Run analysis
            ana_dat <- analyze_image(image,mask_path)
            img_dat <- rbind(img_dat,ana_dat)
        }
        print('Mask extration and data complete. Writing file')
        csv_path <- paste0(video_data_dir,'data.csv')
        write.table(img_dat,file = csv_path)
        print('Data extracted from image')
        return(img_dat)
    }
    if(create_vid == TRUE){
        dir.create('./dat/vids',showWarnings = FALSE)
        make_analysis_video(img_dat, vid_name,)
    }
    
}

run_train <- function(){
    data <- list()
    train_video <- list.files(train_video_path,'.mp4$')
    for (video in train_video){
        video_path <<- paste0(train_video_path,video)
        print(paste0('Currently on:',video_path))
        out_dat <- run_video(video_path,make_img = TRUE, analyze_img=TRUE,type='train')
        data[[video]] <- out_dat
    }
    return(data)
}

run_test <- function(){
    data <- list()
    test_video <- list.files(test_video_path,'.mp4$')
    for (video in train_video){
        video_path <- paste0(test_video_path,video)
        print(paste0('Currently on:',video_path))
        out_dat <- run_video(video_path,make_img = TRUE, analyze_img=TRUE,'type'=test)
        data[[video]] <- out_dat
    }
    return(data)
}




read_vid_csv <- function(){
    data <- list()
    folders <- list.files(root_train_path)
    for (vid_folder in folders){
        csv_path <- paste0(root_train_path,vid_folder,'/data.csv')
        data_vid <- read.csv(csv_path, sep = ' ') %>% as_tibble
        data[[vid_folder]] <- data_vid
    }
    return(data)
}


get_labels <- function(){
    ## Get all info from JSON and return tibble
    out_dat <- tibble(
        name = character(),
        label = numeric(),
        split = factor(),
        original = character()
    )
    json_path <- paste0(train_video_path,'metadata.json')
    json_1 <<- jsonlite::fromJSON(json_path)
    json_2 <<- lapply(json_1, function(x) {
  x[sapply(x, is.null)] <- NA
  unlist(x)
    })
    for(vid_name in names(json_2)){
        label <- json_2[[vid_name]][['label']]
        split <- json_2[[vid_name]][['split']]
        original <- json_2[[vid_name]][['original']]
        if(label == "FAKE"){
            label = 1
        }
        else if(label == "REAL"){
            label = 0
        }
        dat_vec = tibble(Name = vid_name, Fake = label, Original = original)    
        #dat_vec = tibble(name = vid_name, label = label, split=split, original = original)
        out_dat <- rbind(out_dat,dat_vec)
    }
    out_dat <- out_dat %>% column_to_rownames(var="Name")
    return(out_dat)
}

prepare_train <- function(){
    vid_list <- read_vid_csv()
    labels <- get_labels()
    X <- data.frame(red = numeric(),
                green = numeric(),
                blue=numeric(),
                tot_color=numeric(),
                size=numeric()) %>% as_tibble
    y <- data.frame(Label=numeric(0)) %>% as_tibble
    for(vid_name in names(vid_list)){

        tot_name <- paste0(vid_name,'.mp4')
        # Switch this out with process step
        max <- vid_list[[vid_name]] %>% as_tibble %>% summarize(red=max(red),green=max(green),blue=max(blue),tot_color=max(tot_color),size = max(size))
        X <- rbind(X,max)
        label <- labels[tot_name,]['Fake'] %>% as_tibble
        y <- rbind(y,label)
    }
    MM <- bind_cols(X,y)
    return(MM)
}

train_nn_bin <- function(X,y){
    X <- X %>% scale
    y <- y %>% to_categorical

    
    model <- keras_model_sequential() 
    model %>% 
        layer_dense(units = 256, activation = 'relu', input_shape = ncol(X)) %>% 
        layer_dropout(rate = 0.4) %>% 
        layer_dense(units = 128, activation = 'relu') %>%
        layer_dropout(rate = 0.3) %>%
        layer_dense(units = 2, activation = 'sigmoid')

history <- model %>% compile(
                         loss = 'binary_crossentropy',
                         optimizer = 'adam',
                         metrics = c('accuracy')
                     )
    
model %>% fit(
              X, y, 
              epochs = 100, 
              batch_size = 5,
              validation_split = 0.3
          )

return(model)
}
