int n_parts_x;
int n_parts_y;

/******************************************
 * Filter 1 cpu
 * ***************/
void transposeBank(float* &filter_bank) {
	float * transposeBank(float* filter_bank) {
	    // reorganize data in SIMD8 vectors
	    // |0 1 2 .. 8| 0 1 2 .. 8 ..  =>> 0 0 0 ... 1 1 1 ..
	    int filter_size = 9;
	    int num_filters = 100;
	    float* tmpbank = (float*)malloc(num_filters * filter_size*sizeof(float)); // __malloc()
	    for(int i=0; i<num_filters/8; i++)
	    {
	        for(int j=0; j<9; j++) {
	            for(int k=0; k<8; k++)
	                tmpbank[i*8*9+ j*8+ k] = filter_bank[i*8*9+ j+ k*9];
	        }
	    }
	    // leftovers in smaller vecs

	    {
	        for(int j=0; j<9; j++) {
	            for(int k=0; k<4; k++)
	                tmpbank[96*9 + j*4+ k] = filter_bank[96*9 + j+ k*9];
	        }
	    }
	    return tmpbank;
	}
//-----------------------------------------------------------------
// Optimized Filter 1 that works with a transposed bank of filters
void cosine_filter_transpose(float* fr_data, float* fb_array_main, const int height, const int width, const int filter_h, const int filter_w, const int n_filters, float* ind, float *val, int pitch)
{
    float * fb_array = transposeBank(fb_array_main);
    //do convolution
    const int apron_y = filter_h / 2;
    const int apron_x = filter_w / 2;

    const int filter_size = filter_h * filter_w;

    const int filter_bank_size = filter_size * n_filters;

    int *pixel_offsets=(int*) malloc(sizeof(int)*filter_size);

    int oi = 0;
    for (int ii=-apron_y; ii<=apron_y; ii++) {
        for (int jj=-apron_y; jj<=apron_y; jj++) {
            pixel_offsets[oi] = ii * width + jj;
            oi++;
        }
    }
    // 100 filters, each 9 values
    int n_threads = 1;

    int valid_height = height - 2 * apron_y;
    int height_step = valid_height / n_threads + 1;

    for (int tid=0; tid<n_threads; tid++) {
        int start_y = apron_y + tid * height_step;
        int end_y = min(start_y + height_step, height - apron_y);

        if(ViVidPipeline::run_MG())
        {
            //-------------------------------run MG

            ViVidPipeline::arena_parallel_for( start_y, end_y, 1, [&]  (size_t i) {
                float *image_cache=(float*) malloc(sizeof(float)*filter_size);

                float* fr_ptr = fr_data + i * width + apron_x;
                /*
                float* ass_out = out_data + i * width + apron_x;
                float* wgt_out = ass_out + height * width;
                * */
                float* ass_out = ind + i * pitch/sizeof(float) + apron_x;   // modified to get output in two separated arrays
                float* wgt_out = val + i * pitch/sizeof(float) + apron_x;


                for (int j=apron_x; j<(width - apron_x); j++ ) {


                    for (int ii=0; ii< filter_size; ii++) {
                        // copy each pixel to all elements of vector
                        image_cache[ii] = fr_ptr[pixel_offsets[ii]];
                    }

                    float max_sim = -1e6;
                    int best_ind = -1;

                    int fi=0;
                    int filter_ind = 0;
                    int sssize = 9;
                    // 96 filters, 9 values each
                    while (fi<((n_filters/8)*8)*filter_size)
                    {
                        float temp_sum[8] = {0,0,0,0,0,0,0,0};


                        for(int i=0; i<sssize; i++) {

                            //__assume_aligned(fb_array,32);
                            //__assume_aligned(image_cache,32);
                            float img = image_cache[i];
                            for(int j=0; j<8; j++) {
                                temp_sum[j] += img*fb_array[fi++];
                            }
                        }

                        for(int j=0; j<8; j++) {
                            temp_sum[j] = abs(temp_sum[j]);
                        }

                        for(int j=0; j<8; j++) {
                            if(temp_sum[j] > max_sim) {
                                max_sim = temp_sum[j];
                                best_ind = filter_ind+j;
                            }
                        }

                        filter_ind += 8;
                    }

                    float temp_sum[4] = {0,0,0,0};

                    for(int i=0; i<9; i++) {
                        //__assume_aligned(fb_array,32);
                        //__assume_aligned(image_cache,32);
                        //#pragma ivdep
                        for(int j=0; j<4; j++) {
                            temp_sum[j] += image_cache[i]*fb_array[fi++];
                        }
                    }

                    for(int j=0; j<4; j++) {
                        temp_sum[j] = abs(temp_sum[j]);
                    }
                    for(int j=0; j<4; j++) {
                        if(temp_sum[j] > max_sim) {
                            max_sim = temp_sum[j];
                            best_ind = filter_ind+j;
                        }
                    }

                    *ass_out = (float)best_ind;
                    *wgt_out = max_sim;
                    //printf("max_sim %d\n",(int)max_sim);

                    fr_ptr++;
                    ass_out++;
                    wgt_out++;

                }
                free(image_cache);  // same thing
            } );


        } else {
            //-------------------------------run CG
            float *image_cache=(float*) malloc(sizeof(float)*filter_size);

            for (int i=start_y; i<end_y; i++) {
                float* fr_ptr = fr_data + i * width + apron_x;
                /*
                float* ass_out = out_data + i * width + apron_x;
                float* wgt_out = ass_out + height * width;
                * */
                float* ass_out = ind + i * pitch/sizeof(float) + apron_x;   // modified to get output in two separated arrays
                float* wgt_out = val + i * pitch/sizeof(float) + apron_x;


                for (int j=apron_x; j<(width - apron_x); j++ ) {


                    for (int ii=0; ii< filter_size; ii++) {
                        // copy each pixel to all elements of vector
                        image_cache[ii] = fr_ptr[pixel_offsets[ii]];
                    }

                    float max_sim = -1e6;
                    int best_ind = -1;

                    int fi=0;
                    int filter_ind = 0;
                    int sssize = 9;
                    // 96 filters, 9 values each
                    while (fi<((n_filters/8)*8)*filter_size)
                    {
                        float temp_sum[8] = {0,0,0,0,0,0,0,0};


                        for(int i=0; i<sssize; i++) {

                            //__assume_aligned(fb_array,32);
                            //__assume_aligned(image_cache,32);
                            float img = image_cache[i];
                            for(int j=0; j<8; j++) {
                                temp_sum[j] += img*fb_array[fi++];
                            }
                        }

                        for(int j=0; j<8; j++) {
                            temp_sum[j] = abs(temp_sum[j]);
                        }

                        for(int j=0; j<8; j++) {
                            if(temp_sum[j] > max_sim) {
                                max_sim = temp_sum[j];
                                best_ind = filter_ind+j;
                            }
                        }

                        filter_ind += 8;
                    }

                    float temp_sum[4] = {0,0,0,0};

                    for(int i=0; i<9; i++) {
                        //__assume_aligned(fb_array,32);
                        //__assume_aligned(image_cache,32);
                        //#pragma ivdep
                        for(int j=0; j<4; j++) {
                            temp_sum[j] += image_cache[i]*fb_array[fi++];
                        }
                    }

                    for(int j=0; j<4; j++) {
                        temp_sum[j] = abs(temp_sum[j]);
                    }
                    for(int j=0; j<4; j++) {
                        if(temp_sum[j] > max_sim) {
                            max_sim = temp_sum[j];
                            best_ind = filter_ind+j;
                        }
                    }

                    *ass_out = (float)best_ind;
                    *wgt_out = max_sim;
                    //printf("max_sim %d\n",(int)max_sim);

                    fr_ptr++;
                    ass_out++;
                    wgt_out++;
                }
            }
            free(image_cache);  // same thing
        }


    }
    free(fb_array);
    free(pixel_offsets); //added by andres, I think it is necessary

}
//------------------------------------

/**************************************
 * Filter 2 cpu
 * *************************/
 float * block_histogram(FloatBuffer *his, FloatBuffer *ind, FloatBuffer *val, int max_bin, int cell_size, int start_x, int start_y, int im_height, int im_width) {

     //variables
     float * id_data = ind->get_HOST_PTR(BUF_READ);
     float * wt_data = val->get_HOST_PTR(BUF_READ);

     n_parts_y = (im_height-2) / cell_size;
     n_parts_x = (im_width-2) / cell_size;
     int start_i = 1;
     int start_j = 1;
     //end variables

 //	float * out_data = (float *) malloc(n_parts_x * n_parts_y * max_bin * sizeof(float)); // _mm_malloc()

     float* ptr_his = his->get_HOST_PTR(BUF_WRITE); //= (float *) malloc(his->size);

     for (int write_i=0; write_i<n_parts_y; write_i++) {
         for (int write_j=0; write_j<n_parts_x; write_j++) {
             int out_ind = (write_i*n_parts_x + write_j) * his->pitch/sizeof(float);
             int read_i = (start_i + (write_i * cell_size)) * ind->pitch/sizeof(float);
             for (int i=0; i<cell_size; i++) {
                 int read_j = start_j + write_j * cell_size ;

                 for (int j=0; j<cell_size; j++) {
                     int bin_ind = (int)id_data[read_i+read_j+j];
                     //andres
                     assert((bin_ind >= 0) && (bin_ind < max_bin));
                     //if(bin_ind<0) {printf("ERROR  bin_ind<0   %d %d %d\n", read_i,read_j,j); exit(-1);}
                     //if(bin_ind<0) {printf("ERROR  bin_ind>max %d > &d \n",bin_ind,max_bin); exit(-1);}

                     float weight = wt_data[read_i+read_j+j];
                     ptr_his[out_ind + bin_ind] += weight;
                 }
                 read_i += ind->pitch/sizeof(float);
             }
         }
     }
     return his->data;
 }


/*****************************************
 * Filter 3 CPU
 * ***************************/
 float * pwdist_c(FloatBuffer* a_data,  FloatBuffer * b_data,  FloatBuffer* out) {

     float *ptra = a_data->get_HOST_PTR(BUF_READ); // should receive de pointer and then use it. now ptra = a_data->data
     float *ptrb = b_data->get_HOST_PTR(BUF_READ);
     float * out_data= out->get_HOST_PTR(BUF_WRITE);

     //cerr << "A: height = " << aheight << " width = " << awidth << " B: height = " << bheight << " width " << bwidth << endl;
     //  float * out_data = out->data;
     int owidth = out->pitch/sizeof(float);
     int aheight=a_data->height;
     int awidth=a_data->pitch/sizeof(float);
     int bheight=b_data->height;
     //printf(" a cla %dx%d  %d\n",a_data->height,a_data->width,a_data->pitch/sizeof(float));
     //printf(" b his %dx%d  %d\n",b_data->height,b_data->width,b_data->pitch/sizeof(float));

     //#pragma warning( disable: 588)
         //-------------------------------run CG
         for (unsigned int i=0; i<aheight; i++) {
             for (unsigned int j=0; j<bheight; j++) {
                 float sum = 0.0;

                 for (int k = 0; k < a_data->width; k++) {

                     float dif = (ptra[i*awidth+k] - ptrb[j*awidth+k]);
                     sum+=dif*dif;
                 }
                 out_data[i*owidth+j]=sum;

             }
         }



     return out_data;
 }
