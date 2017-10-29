int n_parts_x;
int n_parts_y;

/******************************************
 * Filter 1 cpu
 * ***************/
void transposeBank(float* &filter_bank) {
	// reorganize data in SIMD8 vectors
	// |0 1 2 .. 8| 0 1 2 .. 8 ..  =>> 0 0 0 ... 1 1 1 ..
	int filter_size = 9;
	int num_filters = 100;
  float* tmpbank = (float*)malloc(num_filters * filter_size*sizeof(float));

	for(int i=0; i<num_filters/8; i++)
	{
		for(int j=0; j<9; j++) {
			for(int k=0; k<8; k++)
				tmpbank[i*8*9+ j*8+ k] = filter_bank[i*8*9+ j+ k*9];
		}
	}
		for(int j=0; j<9; j++) {
			for(int k=0; k<4; k++)
				tmpbank[96*9 + j*4+ k] = filter_bank[96*9 + j+ k*9];
		}
	free(filter_bank);
	filter_bank = tmpbank;
}
//-----------------------------------------------------------------
// Optimized Filter 1 that works with a transposed bank of filters
void cosine_filter_transpose(float* fr_data, float* fb_array, const int height, const int width, const int filter_h, const int filter_w, const int n_filters, float* ind, float *val)
{
	transposeBank(fb_array);
	int cont=0;
	float* out_data= (float*) malloc(2 * height * width * sizeof(float));
	//do convolution
	const int apron_y = filter_h / 2;
	const int apron_x = filter_w / 2;
	const int filter_size = filter_h * filter_w;
	const int filter_bank_size = filter_size * n_filters;
	int *pixel_offsets=(int*) malloc(sizeof(int)*filter_size);
	int oi = 0;
	for (int ii=-apron_y; ii<=apron_y; ii++){
		for (int jj=-apron_y; jj<=apron_y; jj++){
			pixel_offsets[oi] = ii * width + jj;
			oi++;
		}
	}
	// 100 filters, each 9 values
	int n_threads = 1;
	int valid_height = height - 2 * apron_y;
	int height_step = valid_height / n_threads + 1;
	float *image_cache=(float*) malloc(sizeof(float)*filter_size);

	for (int tid=0; tid<n_threads; tid++){
		int start_y = apron_y + tid * height_step;
		int end_y = std::min(start_y + height_step, height - apron_y);

		for (int i=start_y; i<end_y; i++){
			float* fr_ptr = fr_data + i * width + apron_x;
			float* ass_out = ind + i * width + apron_x;
      float* wgt_out = val + i * width + apron_x;

			for (int j=apron_x; j<(width - apron_x); j++ ){
				for (int ii=0; ii< filter_size; ii++){
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
						float img = image_cache[i];

						for(int j=0; j<8; j++) {
							temp_sum[j] += img*fb_array[fi++];
						}
					}
					for(int j=0; j<8; j++) {
						temp_sum[j] = std::abs(temp_sum[j]);
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
					for(int j=0; j<4; j++) {
						temp_sum[j] += image_cache[i]*fb_array[fi++];
					}
				}
				for(int j=0; j<4; j++) {
					temp_sum[j] = std::abs(temp_sum[j]);
				}
				for(int j=0; j<4; j++) {
					if(temp_sum[j] > max_sim) {
						max_sim = temp_sum[j];
						best_ind = filter_ind+j;
					}
				}
				*ass_out = (float)best_ind;
				*wgt_out = max_sim;

			//	std::cout <<"ass= " <<*ass_out <<"wgt= "<<*wgt_out<< '\n';
				 *ind=*ass_out;//Added by jc, because if not, ind and val are always 0, I think.
				 *val=*wgt_out;

				// std::cout << "ind= "<<*ind << "val= "<<*val<<'\n';
				// std::cout << ind[i] << val[i] << '\n';
				fr_ptr++;
				ass_out++;
				wgt_out++;
				ind++;//Added by jc, because if not, ind and val are always 0, I think.
				val++;





			}
		}
	}

}
//------------------------------------

/**************************************
 * Filter 2 cpu
 * *************************/
float * block_histogram(float * ind,float* val, int max_bin, int cell_size, int start_x, int start_y, int im_height, int im_width){
	//variables
	float * id_data = ind;
	float * wt_data = val;

	// for(int i=0;i<im_height*im_width;i++){
	// //for(int i=0;i<(sizeof(ind)/sizeof(float));i++){
	// std::cout <<"hola"<< *ind << '\n';
	// ind++;
	// }


	n_parts_y = (im_height - 2) / cell_size;
	n_parts_x = (im_width - 2) / cell_size;
	int start_i = 1;
	int start_j = 1;
	//end variables
	float * out_data = (float *) malloc(n_parts_x * n_parts_y * max_bin * sizeof(float));
    for (int write_i=0; write_i<n_parts_y; write_i++){
        for (int write_j=0; write_j<n_parts_x; write_j++){
            int out_ind = (write_i*n_parts_x + write_j) * max_bin;
            int read_i = (start_i + (write_i * cell_size)) * im_width;

            for (int i=0; i<cell_size; i++){
                int read_j = start_j + write_j * cell_size ;

                for (int j=0; j<cell_size; j++){

									int bin_ind = (int)id_data[read_i+read_j+j];
									assert((bin_ind >= 0) && (bin_ind < max_bin));
                  float weight = wt_data[read_i+read_j+j];
                  out_data[out_ind + bin_ind] += weight;
									//std::cout << out_data[out_ind + bin_ind] << '\n';
                }
                read_i += im_width;
            }
        }
    }
	return out_data;
}


/*****************************************
 * Filter 3 CPU
 * ***************************/
float * pwdist_c(float* a_data, int aheight, int awidth, float * b_data, int bheight, int bwidth){
    float * out_data = (float*) malloc(sizeof(float) * aheight * bheight);
		int owidth = bheight;

    for (unsigned int i=0; i<aheight; i++){
        for (unsigned int j=0; j<bheight; j++){
            float sum = 0.0;

            for (int k = 0; k < awidth; k++){
                float dif = (a_data[i*awidth+k] - b_data[j*awidth+k]);
                sum+=dif*dif;
            }
            out_data[i*owidth+j]=sum;
						//std::cout << out_data[i*owidth+j] << '\n';
        }
    }
    return out_data;
}
