/*********************************************************
**********************************************************
* Dept. Computer Architecture at UMA (C)
* MASTER THESIS
* MASTER IN MECHATRONICS ENGINEERING
*
* AUTHOR: JOSE CARLOS ROMERO MORENO
* TUTOR: RAFAEL ASENJO PLAZA
*
* Leveraging the new OpenCL module in Intel TBB
*
*
* Parallelization algorithm "VIVID".

***************************************************************/
#define TBB_PREVIEW_FLOW_GRAPH_NODES 1
#define TBB_PREVIEW_FLOW_GRAPH_FEATURES 1

#include "common/utility/utility.h"
#include "tbb/tbb.h"
#include "tbb/flow_graph.h"
#include "tbb/flow_graph_opencl_node.h"
#include "tbb/task_scheduler_init.h"
using namespace tbb;
using namespace tbb::flow;
#include <cstdio>
#include <cmath>
#include "tbb/tick_count.h"

//Here you can find the 3 CPU's filters
#include "myfilters_old.h"

/****************************
          MAIN
****************************/
int main(int argc, const char* argv[]) {

    int NUM_IMAGES;
    int cont=0;
    int cont_cpu=0;
    int  n=0;
    int ID_imagen=0;
    int height;
    int width;
    int tok_cpu=0;
    int tok_gpu=0;
    int nth=0;
    char* exampleImagePath;
    char* exampleImagePath2;
    float* filter_bank_gpu;
    float * coefficients_gpu;
    int filter_dim=3;
    int num_filters=100;
    int pitch=width*sizeof(float);
    int n_parts_y_gpu = (416-2) / 8;
    int n_parts_x_gpu = (600-2) / 8;
//create a random coeffcients
    const int window_height = 128;
    const int window_width = 64;
    const int cell_size = 8;
    const int block_size = 2;
    const int dict_size = 100;
    int n_cells_x = window_width / cell_size;
    int n_cells_y = window_height / cell_size;
    int n_blocks_x = n_cells_x - block_size + 1;
    int n_blocks_y = n_cells_y - block_size + 1;
    int n_total_coeff = block_size * block_size * n_blocks_x * n_blocks_y * dict_size;
//Buffers required
    typedef opencl_buffer<float> buffer_float;
    typedef tuple<buffer_float,int,int,int,buffer_float,buffer_float,int,buffer_float,int> buffer_datosGPU1;
    typedef tuple<buffer_float,int,buffer_float,int,buffer_float,int,int,int,int> buffer_datosGPU2;
    typedef tuple<buffer_float,int,int,buffer_float,int,buffer_float,int> buffer_datosGPU3;
    typedef tuple<float*,float*,float*,float*,float*,int,int> datos_imagen;
    typedef tuple<buffer_float,int> out_gpu;
    typedef tuple<float*,int> out_cpu;


    if (argc < 4){
  		printf("Not enough parameters.\n 1- NUM_IMAGES.\n 2- tokens_cpu.\n");
      printf(" 3- tokens_gpu.\n");
      printf("They can be: tokens_cpu [0-4], tokens_gpu [0-1] \n");
  		return -1;
  	}
  	NUM_IMAGES = atoi(argv[1]);
    tok_cpu = atoi(argv[2]);
    tok_gpu = atoi(argv[3]);
    nth = atoi(argv[4]);

    if(nth==0){
      nth=tok_cpu+tok_gpu;
    }
    task_scheduler_init init(nth);

//Graph and OpenCL nodes definition
    opencl_graph g;
    opencl_program<> program(g, "Filtros_GPU_copy.cl");
    opencl_node<buffer_datosGPU1> GPU_F1(g, program.get_kernel("blockwise_distance_kernel"));
    opencl_node<buffer_datosGPU2> GPU_F2(g, program.get_kernel("cellHistogramKernel3"));
    opencl_node<buffer_datosGPU3> GPU_F3(g, program.get_kernel("pairwiseDistanceKernel"));
    std::array<unsigned int, 2> range1{412-2,600-2};
    std::array<unsigned int, 2> range2{(412-2)/8,(600-2)/8};
    std::array<unsigned int, 2> range3{(412-2)/8,(600-2)/8};
    GPU_F1.set_range(range1);
    GPU_F2.set_range(range2);
    GPU_F3.set_range(range3);

//Reading the image
    exampleImagePath = strdup("media/image.bin");
    FILE *fbin = fopen(exampleImagePath,"r");
    if(! fbin ) // Check for invalid input
      {
        std::cout <<  "Could not open or find the image " << exampleImagePath << std::endl ;
        return -1;
      }
    fread(&height, sizeof(int), 1, fbin);
    fread(&width, sizeof(int), 1, fbin);
    float* f_imData= new float [height*width];
    fread(f_imData, sizeof(float), height*width, fbin);
    fclose(fbin);

//This is the way to define buffers for the OpenCL node.
    buffer_float opencl_imagen(g,height*width);
    buffer_float opencl_ind(g,height*width);
    buffer_float opencl_val(g,height*width);
    buffer_float opencl_FilterBank(g,num_filters * filter_dim * filter_dim);
    buffer_float opencl_his(g,n_parts_x_gpu * n_parts_y_gpu * 100);
    buffer_float opencl_out(g,n_total_coeff/dict_size*n_parts_x_gpu * n_parts_y_gpu);
    buffer_float opencl_coefficients(g,n_total_coeff);
//Variables for GPU stages
    for (int i = 0; i < num_filters * filter_dim * filter_dim; i++)
      {

        opencl_FilterBank[i]=float( std::rand() ) / RAND_MAX;
      }
    for(int i = 0; i < n_total_coeff; i++)
      {
        opencl_coefficients[i]= float(std::rand())/RAND_MAX;
      }
      for(int i=0;i<height*width;i++)
     {
       opencl_imagen[i]=f_imData[i];
     }

    tbb::tick_count TGPU,TGPU1,TGPU2,TGPU3,TCPU,TCPU1,TCPU2,TCPU3;
/****************************
SOURCE NODE
*****************************/
    source_node<int> input_node(g,[&](int &a)->bool
    {
       if(ID_imagen<NUM_IMAGES)
       {
      		a =ID_imagen;
          ID_imagen++;
      		return true;
  		 } else
         return false;

		},false);

    typedef join_node<tuple<int,int>, reserving > join_t;
    join_t join_input(g);

    buffer_node<int> token_buffer(g);

/****************************
Connection_node. This node creates a path for the CPU and other for the GPU.
*****************************/
    typedef multifunction_node<join_t::output_type, tuple<int,int> > mfn_t1;
    mfn_t1 connection_node(g, unlimited, [&](const join_t::output_type &in, mfn_t1::output_ports_type &ports ) {

      if (get<1>(in) == 1)
      {
      //Go to the CPU, to CPU_F1.
          get<0>(ports).try_put(get<0>(in));
      } else
        {
      //Go to the GPU, to dispatch_node.
          get<1>(ports).try_put(get<0>(in));
        }
    });

/****************************
DISPATCH_NODE. This node starts the stage 1 for GPU and begins the GPU path.
*****************************/
    typedef multifunction_node<int, tuple<buffer_float,int,int,int,buffer_float,buffer_float,int,buffer_float,int,int> > mfn_t;
    mfn_t dispatch_node(g, unlimited, [&](int ID_imagen, mfn_t::output_ports_type &ports ) {

       TGPU = tbb::tick_count::now();
       get<0>(ports).try_put(opencl_imagen);
       get<1>(ports).try_put(width);
       get<2>(ports).try_put(height);
       get<3>(ports).try_put(pitch);
       get<4>(ports).try_put(opencl_ind);
       get<5>(ports).try_put(opencl_val);
       get<6>(ports).try_put(pitch);
       get<7>(ports).try_put(opencl_FilterBank);
       get<8>(ports).try_put(num_filters);
       get<9>(ports).try_put(ID_imagen);

   });
/********************************
*JOIN_NODE_GPU
****************************** */
  	 typedef join_node<buffer_datosGPU1, queueing > join1_t;
     typedef join_node<buffer_datosGPU2, queueing > join2_t;
     typedef join_node<tuple<buffer_float,int,int,buffer_float,int,buffer_float,int,int>, queueing > join3_t;//<buffer_datosGPU3,int>
     join1_t node_joinGPU(g);
     join2_t node_joinGPU2(g);
     join3_t node_joinGPU3(g);

     typedef indexer_node<out_gpu,out_cpu> join4_t;
     join4_t node_out(g);

/********************************
*OUTPUT_GPU1
****************************** */
    typedef multifunction_node<join1_t::output_type, buffer_datosGPU2, reserving > mfn1_t;
    mfn1_t output_gpu1(g, unlimited, [&](const join1_t::output_type &m, mfn1_t::output_ports_type &ports ) {
       //TGPU1 = tbb::tick_count::now();
       //std::cout <<"GPU 1= "<< (TGPU1 - TGPU).seconds() <<" segundos"<<std::endl;
       get<0>(ports).try_put(opencl_his);
       get<1>(ports).try_put(pitch);
       get<2>(ports).try_put(opencl_ind);
       get<3>(ports).try_put(pitch);
       get<4>(ports).try_put(opencl_val);
       get<5>(ports).try_put(pitch);
       get<6>(ports).try_put(num_filters);
       get<7>(ports).try_put(cell_size);
       get<8>(ports).try_put(n_parts_x_gpu);
    });
/********************************
*OUTPUT_GPU2
****************************** */
    typedef multifunction_node<join2_t::output_type, buffer_datosGPU3 > mfn2_t;
    mfn2_t output_gpu2(g, unlimited, [&](const join2_t::output_type &m, mfn2_t::output_ports_type &ports ) {
       //TGPU2 = tbb::tick_count::now();
       //std::cout <<"GPU 2= "<< (TGPU2 - TGPU1).seconds() <<" segundos"<<std::endl;
       get<0>(ports).try_put(opencl_coefficients);
       get<1>(ports).try_put(width);
       get<2>(ports).try_put(width);
       get<3>(ports).try_put(opencl_his);
       get<4>(ports).try_put(width);
       get<5>(ports).try_put(opencl_out);
       get<6>(ports).try_put(width);
    });

/********************************
 *OUTPUT_GPU3
****************************** */
    typedef multifunction_node<join3_t::output_type, tuple<out_gpu,int> > mfn3_t;
    mfn3_t output_gpu3(g, unlimited, [&](const join3_t::output_type &m, mfn3_t::output_ports_type &ports ) {
       //TGPU3 = tbb::tick_count::now();
       int token_gpu=0;
       //std::cout <<"GPU 3= "<< (TGPU3 - TGPU2).seconds() <<" segundos"<<std::endl;
       //std::cout << "Total GPU="<<(TGPU3-TGPU).seconds() <<" segundos"<< '\n';
       cont++;
       out_gpu out_gpu1;
       get<0>(out_gpu1)=get<5>(m);//Image
       get<1>(out_gpu1)=get<7>(m);//ID

       get<0>(ports).try_put(out_gpu1);//Image+ID
       get<1>(ports).try_put(token_gpu);//Token
    });

/********************************
 *CPU_STAGES
****************************** */

	function_node<int,datos_imagen> cpu_f1(g,unlimited,[&](int ID_imagen) {

    datos_imagen datos2;
    get<6>(datos2)=ID_imagen;
    float* image2= new float [height*width];
    memcpy(image2,f_imData,height*width);
    get<0>(datos2)=image2;
    get<5>(datos2)=1;
    get<1>(datos2)=new float [height*width];
    get<2>(datos2)=new float [height*width];

    float* filter_bank = new float [num_filters * filter_dim * filter_dim ];
		for (int i = 0; i < num_filters * filter_dim * filter_dim; i++)
			{
				filter_bank[i] = float( std::rand() ) / RAND_MAX;
			}
	  cosine_filter_transpose(get<0>(datos2),filter_bank,height,width,filter_dim,filter_dim,num_filters,get<1>(datos2),get<2>(datos2));
		return datos2;
	});

	function_node<datos_imagen,datos_imagen> cpu_f2(g,unlimited,[&](datos_imagen datos) {

      float* his = block_histogram(get<1>(datos),get<2>(datos), num_filters, 8, 0, 0, height, width);
			get<3>(datos)=his;
			delete(get<1>(datos));
			delete(get<2>(datos));
      return datos;
    });

  function_node<datos_imagen,datos_imagen> cpu_f3(g,unlimited,[&](datos_imagen datos) {

      float * coefficients = new float [n_total_coeff];
			for(int i = 0; i < n_total_coeff; i++)
        {
  				coefficients[i] = float(std::rand())/RAND_MAX;
  			}
      float *results = pwdist_c(coefficients, n_total_coeff/dict_size, dict_size, get<3>(datos), n_parts_x * n_parts_y , num_filters );
			get<4>(datos)=results;
			delete(get<3>(datos));
			delete(get<4>(datos));
      return datos;
   });

  typedef multifunction_node<datos_imagen, tuple<out_cpu,int> > mfn5_t;
  mfn5_t output_cpu(g, unlimited, [&](datos_imagen datos, mfn5_t::output_ports_type &ports ) {
      cont_cpu++;
      out_cpu out_cpu1;
      get<0>(out_cpu1)=get<4>(datos);//Image
      get<1>(out_cpu1)=get<6>(datos);//ID

      get<0>(ports).try_put(out_cpu1);//Image+ID
      get<1>(ports).try_put(get<5>(datos));//Token
   });

/********************************
 *OUTPUT_DISPLAY
****************************** */
    function_node<join4_t::output_type> output_display(g, unlimited, [&](const join4_t::output_type &m){
      if(m.tag()==0){
        //std::cout << "gpu" << '\n';
        out_gpu gpu=cast_to<out_gpu>(m);
        //std::cout << get<1>(gpu) << '\n';

      }else{
        //std::cout << "cpu" << '\n';
        out_cpu cpu=cast_to<out_cpu>(m);
        //std::cout << get<1>(cpu) << '\n';
      }

    });

/********************************
Edges from source_node to dispatch_node and CPU stages
********************************/
    make_edge(input_node,input_port<0>(join_input));
    make_edge(token_buffer,input_port<1>(join_input));
    make_edge(join_input,connection_node);
    make_edge(output_port<0>(connection_node), cpu_f1);

    make_edge(cpu_f1,cpu_f2);
    make_edge(cpu_f2,cpu_f3);
    make_edge(cpu_f3,output_cpu);
    make_edge(output_port<0>(output_cpu),input_port<1>(node_out));
    make_edge(output_port<1>(output_cpu),token_buffer);

/********************************
Edges from Node 1 GPU
********************************/
    make_edge(output_port<1>(connection_node), dispatch_node);

    make_edge(output_port<0>(dispatch_node),input_port<0>(GPU_F1));
    make_edge(output_port<1>(dispatch_node),input_port<1>(GPU_F1));
    make_edge(output_port<2>(dispatch_node),input_port<2>(GPU_F1));
    make_edge(output_port<3>(dispatch_node),input_port<3>(GPU_F1));
    make_edge(output_port<4>(dispatch_node),input_port<4>(GPU_F1));
    make_edge(output_port<5>(dispatch_node),input_port<5>(GPU_F1));
    make_edge(output_port<6>(dispatch_node),input_port<6>(GPU_F1));
    make_edge(output_port<7>(dispatch_node),input_port<7>(GPU_F1));
    make_edge(output_port<8>(dispatch_node),input_port<8>(GPU_F1));

    make_edge(output_port<0>(GPU_F1),input_port<0>(node_joinGPU));
    make_edge(output_port<1>(GPU_F1),input_port<1>(node_joinGPU));
    make_edge(output_port<2>(GPU_F1),input_port<2>(node_joinGPU));
    make_edge(output_port<3>(GPU_F1),input_port<3>(node_joinGPU));
    make_edge(output_port<4>(GPU_F1),input_port<4>(node_joinGPU));
    make_edge(output_port<5>(GPU_F1),input_port<5>(node_joinGPU));
    make_edge(output_port<6>(GPU_F1),input_port<6>(node_joinGPU));
    make_edge(output_port<7>(GPU_F1),input_port<7>(node_joinGPU));
    make_edge(output_port<8>(GPU_F1),input_port<8>(node_joinGPU));

    make_edge(node_joinGPU,output_gpu1);

/********************************
Edges from Node 2 GPU
********************************/
    make_edge(output_port<0>(output_gpu1),input_port<0>(GPU_F2));
    make_edge(output_port<1>(output_gpu1),input_port<1>(GPU_F2));
    make_edge(output_port<2>(output_gpu1),input_port<2>(GPU_F2));
    make_edge(output_port<3>(output_gpu1),input_port<3>(GPU_F2));
    make_edge(output_port<4>(output_gpu1),input_port<4>(GPU_F2));
    make_edge(output_port<5>(output_gpu1),input_port<5>(GPU_F2));
    make_edge(output_port<6>(output_gpu1),input_port<6>(GPU_F2));
    make_edge(output_port<7>(output_gpu1),input_port<7>(GPU_F2));
    make_edge(output_port<8>(output_gpu1),input_port<8>(GPU_F2));

    make_edge(output_port<0>(GPU_F2),input_port<0>(node_joinGPU2));
    make_edge(output_port<1>(GPU_F2),input_port<1>(node_joinGPU2));
    make_edge(output_port<2>(GPU_F2),input_port<2>(node_joinGPU2));
    make_edge(output_port<3>(GPU_F2),input_port<3>(node_joinGPU2));
    make_edge(output_port<4>(GPU_F2),input_port<4>(node_joinGPU2));
    make_edge(output_port<5>(GPU_F2),input_port<5>(node_joinGPU2));
    make_edge(output_port<6>(GPU_F2),input_port<6>(node_joinGPU2));
    make_edge(output_port<7>(GPU_F2),input_port<7>(node_joinGPU2));
    make_edge(output_port<8>(GPU_F2),input_port<8>(node_joinGPU2));

    make_edge(node_joinGPU2,output_gpu2);

/********************************
Edges from Node 3 GPU
********************************/
    make_edge(output_port<0>(output_gpu2),input_port<0>(GPU_F3));
    make_edge(output_port<1>(output_gpu2),input_port<1>(GPU_F3));
    make_edge(output_port<2>(output_gpu2),input_port<2>(GPU_F3));
    make_edge(output_port<3>(output_gpu2),input_port<3>(GPU_F3));
    make_edge(output_port<4>(output_gpu2),input_port<4>(GPU_F3));
    make_edge(output_port<5>(output_gpu2),input_port<5>(GPU_F3));
    make_edge(output_port<6>(output_gpu2),input_port<6>(GPU_F3));

    make_edge(output_port<0>(GPU_F3),input_port<0>(node_joinGPU3));
    make_edge(output_port<1>(GPU_F3),input_port<1>(node_joinGPU3));
    make_edge(output_port<2>(GPU_F3),input_port<2>(node_joinGPU3));
    make_edge(output_port<3>(GPU_F3),input_port<3>(node_joinGPU3));
    make_edge(output_port<4>(GPU_F3),input_port<4>(node_joinGPU3));
    make_edge(output_port<5>(GPU_F3),input_port<5>(node_joinGPU3));
    make_edge(output_port<6>(GPU_F3),input_port<6>(node_joinGPU3));

    make_edge(output_port<9>(dispatch_node),input_port<7>(node_joinGPU3));
    make_edge(node_joinGPU3,output_gpu3);

    make_edge(output_port<0>(output_gpu3),input_port<0>(node_out));
    make_edge(node_out,output_display);
    make_edge(output_port<1>(output_gpu3),token_buffer);

/********************************
Beggining of the program and timing
********************************/
    printf("Beginning...\n");
    tbb::tick_count mainStartTime = tbb::tick_count::now();
    input_node.activate();
    if(tok_cpu>0){
      for(int i=0;i<tok_cpu;i++){
        token_buffer.try_put(1);
      }
    }
    if(tok_gpu>0){
      token_buffer.try_put(0);
    }
    std::cout <<"Strategy: "<< tok_cpu <<" CPU "<< tok_gpu <<" GPU"<< '\n';

/********************************
Wait until all nodes end, later the program ends
********************************/
    g.wait_for_all();
    std::cout << "Frames computed by CPU= " <<cont_cpu<< '\n';
    std::cout << "Frames computed by GPU= " <<cont<< '\n';
    std::cout << "Total frames= "<<ID_imagen<< '\n';
    std::cout <<"Execution time= "<< (tbb::tick_count::now() - mainStartTime).seconds() <<" seconds"<<std::endl;
    return 0;
}
