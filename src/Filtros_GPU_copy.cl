/*
    Copyright (c) 2017 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/
/*
__constant int f_width=600;
__constant int f_height=416;
__constant int f_pitch=600/sizeof(float);
__constant int o_pitch=600/sizeof(float);
__constant int n_filters= 100;

__constant int histogram_pitch=600/sizeof(float);
__constant int assignments_pitch=600/sizeof(float);
__constant int weights_pitch=600/sizeof(float);
__constant int n_blocks_x=(600-2) / 8;
__constant int cell_size=8;
__constant int max_bin=100;

__constant int a_width=600;
__constant int a_pitch_f=600/sizeof(float);
__constant int b_pitch_f=600/sizeof(float);
__constant int o_pitch_f=600/sizeof(float);
*/

/****************************************************************************************

ETAPA 1 DE LA GPU

*****************************************************************************************/


__kernel void blockwise_distance_kernel(
    __global float * frame, const int f_width, const int f_height, const int f_pitch,
    __global float * ind,
    __global float * val, const int o_pitch,
    __global float4 * c_FilterBank, const int n_filters)
{



    const int posy = get_global_id(0);
	const int posx = get_global_id(1);


  for(int i=0;i<10;i++){

  //printf("Hola\n");

  }
	const int f_pitch_f = f_pitch / sizeof(float);


	float4 img0 = (float4) frame[ posy    * f_pitch_f + posx];
	float4 img1 = (float4) frame[ posy    * f_pitch_f + posx +1];
	float4 img2 = (float4) frame[ posy    * f_pitch_f + posx +2];
	float4 img3 = (float4) frame[(posy+1) * f_pitch_f + posx];
	float4 img4 = (float4) frame[(posy+1) * f_pitch_f + posx +1];
	float4 img5 = (float4) frame[(posy+1) * f_pitch_f + posx +2];
	float4 img6 = (float4) frame[(posy+2) * f_pitch_f + posx];
	float4 img7 = (float4) frame[(posy+2) * f_pitch_f + posx +1];
	float4 img8 = (float4) frame[(posy+2) * f_pitch_f + posx +2];


	float curval =-1e6f;
    float curid = -1;
    int fi=0;



#define __UNROLL

    // desenrollar 25 x 4 = 100, vectorizado de 4 en 4
#ifdef __UNROLL


// partial unroll beacuse total unrolls leads to impossible to build cl kernel, maybe too big kernel ?

float4 tempval;
int filter_id=0;

for (int iter_id=0; iter_id<5; iter_id++)  //  n_filters/4/5
	{


tempval= (float4)(0.0f);
tempval += c_FilterBank[fi++] * img0;
tempval += c_FilterBank[fi++] * img1;
tempval += c_FilterBank[fi++] * img2;
tempval += c_FilterBank[fi++] * img3;
tempval += c_FilterBank[fi++] * img4;
tempval += c_FilterBank[fi++] * img5;
tempval += c_FilterBank[fi++] * img6;
tempval += c_FilterBank[fi++] * img7;
tempval += c_FilterBank[fi++] * img8;
tempval = fabs(tempval);

if ((tempval.s0) > curval) { curid = iter_id*20+0; curval = (tempval.s0); }
if ((tempval.s1) > curval) { curid = iter_id*20+1; curval = (tempval.s1); }
if ((tempval.s2) > curval) { curid = iter_id*20+2; curval = (tempval.s2); }
if ((tempval.s3) > curval) { curid = iter_id*20+3; curval = (tempval.s3); }

tempval= (float4)(0.0f);
tempval += c_FilterBank[fi++] * img0;
tempval += c_FilterBank[fi++] * img1;
tempval += c_FilterBank[fi++] * img2;
tempval += c_FilterBank[fi++] * img3;
tempval += c_FilterBank[fi++] * img4;
tempval += c_FilterBank[fi++] * img5;
tempval += c_FilterBank[fi++] * img6;
tempval += c_FilterBank[fi++] * img7;
tempval += c_FilterBank[fi++] * img8;
tempval = fabs(tempval);

if ((tempval.s0) > curval) { curid = iter_id*20+4;	curval = (tempval.s0); }
if ((tempval.s1) > curval) { curid = iter_id*20+5;	curval = (tempval.s1); }
if ((tempval.s2) > curval) { curid = iter_id*20+6;	curval = (tempval.s2); }
if ((tempval.s3) > curval) { curid = iter_id*20+7;   curval = (tempval.s3); }

tempval= (float4)(0.0f);
tempval += c_FilterBank[fi++] * img0;
tempval += c_FilterBank[fi++] * img1;
tempval += c_FilterBank[fi++] * img2;
tempval += c_FilterBank[fi++] * img3;
tempval += c_FilterBank[fi++] * img4;
tempval += c_FilterBank[fi++] * img5;
tempval += c_FilterBank[fi++] * img6;
tempval += c_FilterBank[fi++] * img7;
tempval += c_FilterBank[fi++] * img8;
tempval = fabs(tempval);

if ((tempval.s0) > curval) { curid = iter_id*20+8;	curval = (tempval.s0); }
if ((tempval.s1) > curval) { curid = iter_id*20+9;	curval = (tempval.s1); }
if ((tempval.s2) > curval) { curid = iter_id*20+10; curval = (tempval.s2); }
if ((tempval.s3) > curval) { curid = iter_id*20+11; curval = (tempval.s3); }

tempval= (float4)(0.0f);
tempval += c_FilterBank[fi++] * img0;
tempval += c_FilterBank[fi++] * img1;
tempval += c_FilterBank[fi++] * img2;
tempval += c_FilterBank[fi++] * img3;
tempval += c_FilterBank[fi++] * img4;
tempval += c_FilterBank[fi++] * img5;
tempval += c_FilterBank[fi++] * img6;
tempval += c_FilterBank[fi++] * img7;
tempval += c_FilterBank[fi++] * img8;
tempval = fabs(tempval);

if ((tempval.s0) > curval) { curid = iter_id*20+12; curval = (tempval.s0); }
if ((tempval.s1) > curval) { curid = iter_id*20+13; curval = (tempval.s1); }
if ((tempval.s2) > curval) { curid = iter_id*20+14; curval = (tempval.s2); }
if ((tempval.s3) > curval) { curid = iter_id*20+15; curval = (tempval.s3); }


tempval= (float4)(0.0f);
tempval += c_FilterBank[fi++] * img0;
tempval += c_FilterBank[fi++] * img1;
tempval += c_FilterBank[fi++] * img2;
tempval += c_FilterBank[fi++] * img3;
tempval += c_FilterBank[fi++] * img4;
tempval += c_FilterBank[fi++] * img5;
tempval += c_FilterBank[fi++] * img6;
tempval += c_FilterBank[fi++] * img7;
tempval += c_FilterBank[fi++] * img8;
tempval = fabs(tempval);

if ((tempval.s0) > curval) { curid = iter_id*20+16; curval = (tempval.s0); }
if ((tempval.s1) > curval) { curid = iter_id*20+17; curval = (tempval.s1); }
if ((tempval.s2) > curval) { curid = iter_id*20+18; curval = (tempval.s2); }
if ((tempval.s3) > curval) { curid = iter_id*20+19; curval = (tempval.s3); }

}

#else

	for (int filter_id=0; filter_id<n_filters/4; filter_id++)
	{
				    int o_filter_id = filter_id*4;

                    float4 tempval= (float4)(0.0f);

						tempval += c_FilterBank[fi++] * img0;
						tempval += c_FilterBank[fi++] * img1;
						tempval += c_FilterBank[fi++] * img2;


						tempval += c_FilterBank[fi++] * img3;
						tempval += c_FilterBank[fi++] * img4;
						tempval += c_FilterBank[fi++] * img5;


						tempval += c_FilterBank[fi++] * img6;
						tempval += c_FilterBank[fi++] * img7;
						tempval += c_FilterBank[fi++] * img8;



				tempval = fabs(tempval);



				if ((tempval.s0) > curval){
                        curid = o_filter_id;
                        curval = (tempval.s0);
                    }
				if ((tempval.s1) > curval){
                        curid = o_filter_id+1;
                        curval = (tempval.s1);
                    }
				if ((tempval.s2) > curval){
                        curid = o_filter_id+2;
                        curval = (tempval.s2);
                    }
				if ((tempval.s3) > curval){
                        curid = o_filter_id+3;
                        curval = (tempval.s3);
                    }

          }

#endif
/***********************
 * write output values
 *************************/


 const int o_pos=(posy+1)*o_pitch/sizeof(float)+posx+1;  // salvar el borde +1
    ind[o_pos]=curid;
    val[o_pos]=curval;

    for(int i=0;i<10;i++){
      //printf("%f %f \n",ind[i],val[i]);
    }


   //barrier(CLK_GLOBAL_MEM_FENCE); // why? all threads will end here no need to explicit barrier...

}

/****************************************************************************************

ETAPA 2 DE LA GPU

*****************************************************************************************/



// assignments=ind;
//weights=val;

__kernel void cellHistogramKernel3(
    __global float* histogram,
 	const int histogram_pitch,
	__global float* assignments,
	const int assignments_pitch,
	__global float* weights,
	const int weights_pitch,
    const int max_bin,
    const int cell_size,
	const int n_blocks_x
    )
{
    const int block_x =get_global_id(1);
    const int block_y =get_global_id(0);
	const int pix_y = block_y *cell_size + 1 ;
    const int pix_x = block_x *cell_size + 1 ;

    const int histogram_pitch_f = histogram_pitch / sizeof(float);
    const int assignments_pitch_f = assignments_pitch / sizeof(float);
	const int weights_pitch_f = weights_pitch / sizeof(float);


	for(int i = 0; i< cell_size; i++)
	{
	for(int j = 0; j< cell_size; j++)
	{

    const float aval = assignments[(pix_y+j )* assignments_pitch_f + pix_x+i];

    const float wval = weights[(pix_y+j)* weights_pitch_f + pix_x+i];

    const int block= block_y*n_blocks_x+block_x;
    histogram[ block*histogram_pitch_f + (int)aval ]+= wval;  //should be pos, but ...
	}

	}




    /*

	const int histogram_pitch_f = histogram_pitch / sizeof(float);

    const int assignments_pitch_f = assignments_pitch / sizeof(float);
	const int weights_pitch_f = weights_pitch / sizeof(float);

    const int cb_ind_y = get_local_id(0) / 8;
    const int cb_ind_x = get_local_id(1)  / 8;

    const int tc_ind_y = get_local_id(0) % 8;
    const int tc_ind_x = get_local_id(1)  % 8;

    const int target_y = get_group_id(0) * 2 + cb_ind_y;
    const int target_x = get_group_id(1) * 2 + cb_ind_x;

    const int source_y = start_y + BLOCK_SIZE * get_group_id(0) + get_local_id(0);
    const int source_x = start_x + BLOCK_SIZE * get_group_id(1) + get_local_id(1);

    const float aval = assignments[source_y * assignments_pitch_f + source_x];

    const float wval = weights[source_y * weights_pitch_f + source_x];

    const int cells_per_block_dim = 2;

    __local float histogram_cache[2000]; // era local


    const int cache_offset = MAX_HISTOGRAM_SIZE *
        (cb_ind_y * cells_per_block_dim + cb_ind_x);

    //initialize the histogram
    int thread_bin_offset = tc_ind_y * cell_size + tc_ind_x;
    while (thread_bin_offset < max_bin)
    {
        const int cache_addr = cache_offset + thread_bin_offset;
        histogram_cache[cache_addr] = 0;
        thread_bin_offset += (cell_size * cell_size);
    }

   barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    //if (wval > 0.01f){
    //    atomicAdd(histogram_cache + cache_offset + (int) aval, wval);
			histogram_cache [cache_offset] = histogram_cache [cache_offset] + wval;

    //}

   barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    //if ((target_y < histogram.dim_t) && (target_x < histogram.dim_x))
    {
        thread_bin_offset = tc_ind_y * cell_size + tc_ind_x;
        while (thread_bin_offset < max_bin)
        {
            const int cache_addr = cache_offset + thread_bin_offset;

			histogram[(target_y * n_blocks_x + target_x) * histogram_pitch_f + thread_bin_offset] = histogram_cache[cache_addr];
            thread_bin_offset += (cell_size * cell_size);
        }
    }


	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
	* */
}

/****************************************************

ETAPA 3 DE LA GPU

****************************************************/



__kernel void pairwiseDistanceKernel(
        __global float4 * a, const int a_width, const int a_pitch_f,
        __global float4 * b, const int b_pitch_f,
        __global float* out, const int o_pitch_f)
{



	const int a_id = get_global_id(0) ;
    const int b_id = get_global_id(1) ;



    float4 dst = 0;
    float4 diff;
    const int posa = a_id * a_pitch_f/4;
    const int posb = b_id * b_pitch_f/4;

#define __UNROLL

    // desenrollar 25 x 4 = 100, vectorizado de 4 en 4
#ifdef __UNROLL


    diff = a[posa  + 0] - b[posb + 0];
	dst=mad(diff, diff, dst);
	diff = a[posa  + 1] - b[posb + 1];
	dst=mad(diff, diff, dst);
	diff = a[posa  + 2] - b[posb + 2];
	dst=mad(diff, diff, dst);
	diff = a[posa  + 3] - b[posb + 3];
	dst=mad(diff, diff, dst);
    diff = a[posa  + 4] - b[posb + 4];
	dst=mad(diff, diff, dst);
    diff = a[posa  + 5] - b[posb + 5];
	dst=mad(diff, diff, dst);
    diff = a[posa  + 6] - b[posb + 6];
	dst=mad(diff, diff, dst);
    diff = a[posa  + 7] - b[posb + 7];
	dst=mad(diff, diff, dst);
    diff = a[posa  + 8] - b[posb + 8];
	dst=mad(diff, diff, dst);
    diff = a[posa  + 9] - b[posb + 9];
	dst=mad(diff, diff, dst);
    diff = a[posa  + 10] - b[posb + 10];
	dst=mad(diff, diff, dst);
    diff = a[posa  + 11] - b[posb + 11];
	dst=mad(diff, diff, dst);
    diff = a[posa  + 12] - b[posb + 12];
	dst=mad(diff, diff, dst);
    diff = a[posa  + 13] - b[posb + 13];
	dst=mad(diff, diff, dst);
    diff = a[posa  + 14] - b[posb + 14];
	dst=mad(diff, diff, dst);
    diff = a[posa  + 15] - b[posb + 15];
	dst=mad(diff, diff, dst);
    diff = a[posa  + 16] - b[posb + 16];
	dst=mad(diff, diff, dst);
    diff = a[posa  + 17] - b[posb + 17];
	dst=mad(diff, diff, dst);
    diff = a[posa  + 18] - b[posb + 18];
	dst=mad(diff, diff, dst);
    diff = a[posa  + 19] - b[posb + 19];
	dst=mad(diff, diff, dst);
    diff = a[posa  + 20] - b[posb + 20];
	dst=mad(diff, diff, dst);
    diff = a[posa  + 21] - b[posb + 21];
	dst=mad(diff, diff, dst);
    diff = a[posa  + 22] - b[posb + 22];
	dst=mad(diff, diff, dst);
    diff = a[posa  + 23] - b[posb + 23];
	dst=mad(diff, diff, dst);
    diff = a[posa  + 24] - b[posb + 24];
	dst=mad(diff, diff, dst);

#else

    for (int filter_id=0; filter_id<a_width/4; filter_id++)
	{
		diff = a[posa+ filter_id] - b[posb + filter_id];
		dst=mad(diff, diff, dst);
        //dst += diff * diff;
	}

#endif


	out[a_id * o_pitch_f + b_id] = dot(dst,(float4)(1,1,1,1));
}



/*



/// ==================================================================


    const int out_ry = get_group_id(0) * BLOCK_SIZE + get_local_id(1);
    const int out_cx = get_group_id(1) * BLOCK_SIZE + get_local_id(0);

    const int b_ry = get_group_id(1) * BLOCK_SIZE + get_local_id(0);

const int a_mul = out_ry * a_pitch_f/8;

const int b_mul = b_ry * b_pitch_f/8;


    float8 dst = 0;


    for (int i=0; i < a_width/8; i+=BLOCK_SIZE)
    {
        int a_ind = a_mul+i;
int b_ind = b_mul+i;



float8 diff0 = a[a_ind + 0] - b[b_ind + 0];
dst=mad(diff0, diff0, dst);
        //dst += diff0 * diff0;

float8 diff1 = a[a_ind + 1] - b[b_ind + 1];
dst= mad(diff1, diff1, dst);
        //dst += diff1 * diff1;

float8 diff2 = a[a_ind + 2] - b[b_ind + 2];
dst=mad(diff2, diff2, dst);
        //dst += diff2 * diff2;

float8 diff3 = a[a_ind + 3] - b[b_ind + 3];
dst=mad(diff3, diff3, dst);
        //dst += diff3 * diff3;

float8 diff4 = a[a_ind + 4] - b[b_ind + 4];
dst=mad(diff4, diff4, dst);
        //dst += diff4 * diff4;

float8 diff5 = a[a_ind + 5] - b[b_ind + 5];
dst=mad(diff5, diff5, dst);
        //dst += diff5 * diff5;

float8 diff6 = a[a_ind + 6] - b[b_ind + 6];
dst=mad(diff6, diff6, dst);
        //dst += diff6 * diff6;

float8 diff7 = a[a_ind + 7] - b[b_ind + 7];
dst=mad(diff7, diff7, dst);
        //dst += diff7 * diff7;

float8 diff8 = a[a_ind + 8] - b[b_ind + 8];
dst=mad(diff8, diff8, dst);
        //dst += diff8 * diff8;

float8 diff9 = a[a_ind + 9] - b[b_ind + 9];
dst=mad(diff9, diff9, dst);
        //dst += diff9 * diff9;

float8 diff10 = a[a_ind + 10] - b[b_ind + 10];
dst=mad(diff10, diff10, dst);
        //dst += diff10 * diff10;

float8 diff11 = a[a_ind + 11] - b[b_ind + 11];
dst=mad(diff11, diff11, dst);
        //dst += diff11 * diff11;

float8 diff12 = a[a_ind + 12] - b[b_ind + 12];
dst=mad(diff12, diff12, dst);
        //dst += diff12 * diff12;

float8 diff13 = a[a_ind + 13] - b[b_ind + 13];
dst=mad(diff13, diff13, dst);
        //dst += diff13 * diff13;

float8 diff14 = a[a_ind + 14] - b[b_ind + 14];
dst=mad(diff14, diff14, dst);
        //dst += diff14 * diff14;

float8 diff15 = a[a_ind + 15] - b[b_ind + 15];
dst=mad(diff15, diff15, dst);
        //dst += diff15 * diff15;


    }
out[out_ry * o_pitch_f + out_cx] = dot(dst.lo,(float4)(1,1,1,1))+ dot(dst.hi,(float4)(1,1,1,1));
    //out[out_ry * o_pitch_f + out_cx] = dst.s0 +dst.s1 +dst.s2 +dst.s3 +dst.s4 +dst.s5 +dst.s6 +dst.s7;
}
*
* */
