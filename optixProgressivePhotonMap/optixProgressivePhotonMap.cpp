/* 
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//-----------------------------------------------------------------------------
//
// optixProgressivePhotonMap: progressive photon mapping scene
//
//-----------------------------------------------------------------------------


#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

// from sutil
#include <sutil.h>
#include <Camera.h>

#include "Mesh.h"
#include "ppm.h"
#include "random.h"
#include "select.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <limits>
#include <stdint.h>
#include <time.h>

using namespace optix;

const char* const SAMPLE_NAME = "optixProgressivePhotonMap";

unsigned int numero_iteraciones = 1u;
unsigned int numero_hilos       = 4330u;

float scene_epsilon = 1.e-4f;
float beta_diff     = 1.f;
float delta_diff    = beta_diff * 0.5f;

float valorTmin = 300;
float valorTmax = 310;

bool densidad_uniforme = 0u;
bool use_zorder = 0u;

float emisividad = 0.85f; // paper: 0.85
float lambda     = 40;    // conductividad termica
float difusion   = 1.f;   // paper: 1, 0.65, 0.1, 0.9

uint max_path_length = 1500000u;

uint umbral_bbox_hits   = 50u;
uint caminos_generados  = 0u;
uint caminos_cortados   = 0u;
float porcentaje_caminos_cortados   = 0.f;
uint puntos_generados   = 0u;
float ancho_intervalo    = 2.5;

float bbox_x = 80.f;
float bbox_y = 80.f;
float bbox_z = 160.f;

// zorder
uint cant_dimensiones = 5;
uint precision_dir = 2.f;
uint precision = bbox_x;
uint num_cubes = 32;

uint num_intervalos = 64;
uint *Tmin_array_path;
uint *Tmax_array_path;
uint *CYTmin_array_path;
uint *CYTmax_array_path;

std::string bbox_name = "bbox_80x80x160.obj";

std::string object_name = "kelvinCell_paper.obj";

enum SplitChoice {
  RoundRobin,
  HighestVariance,
  LongestDim
};

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context context = 0;

std::ofstream salida;

bool guardar = true;

//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

void printInfo(){
  std::cerr
  << numero_iteraciones  << "_"
  << numero_hilos        << "_"
  << max_path_length     << "_"
  << beta_diff           << "_"
  << delta_diff          << "_"
  << valorTmin           << "_"
  << valorTmax           << "_"
  << "du"       << densidad_uniforme       << "_"
  << "z"        << use_zorder       << "_"
  << emisividad          << "_"
  << lambda              << "_"
  << difusion            << "_"
  << num_intervalos      << "_"
  << umbral_bbox_hits    << "_"
  << bbox_x              << "_"
  << bbox_y              << "_"
  << bbox_z              << "_"
  << caminos_generados   << "_"
  << caminos_cortados    << "_"
  << puntos_generados    << "_"
  << object_name

  << std::endl;
} // printInfo

std::string getFileName(std::string str){
  std::stringstream ss;
  ss
  << lambda           << "_"
  << difusion         << "_"
  << "du"       << densidad_uniforme       << "_"
  << "z"        << use_zorder       << "_"
  << numero_hilos     << "_"
  << object_name      << ""
  << str              << ""
  << ".txt";

  return ss.str();
} // printInfo

std::string getFileNameCyril(std::string str){
  std::stringstream ss;
  ss
  << lambda           << "_"
  << difusion         << "_"
  << "du"       << densidad_uniforme       << "_"
  << "z"        << use_zorder       << "_"
  << numero_hilos     << "_"
  << "cyril"          << "_"
  << object_name      << ""
  << str              << ""
  << ".txt";

  return ss.str();
} // printInfo

/*
int getSPcores(cudaDeviceProp devProp){  
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
     case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
     case 3: // Kepler
      cores = mp * 192;
      break;
     case 5: // Maxwell
      cores = mp * 128;
      break;
     case 6: // Pascal
      if (devProp.minor == 1) cores = mp * 128;
      else if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 7: // Volta
      if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     default:
      printf("Unknown device type\n"); 
      break;
      }
    return cores;
} // getSPcores
*/

static std::string ptxPath( const std::string& cuda_file ){
  return
      std::string(sutil::samplesPTXDir()) +
      "/" + std::string(SAMPLE_NAME) + "_generated_" +
      cuda_file +
      ".ptx";
} // ptxPath

void destroyContext(){

  if( context ){
      context->destroy();
      context = 0;
  } // if

} // destroyContext

// Son Ray generation programs
enum ProgramEnum {
    rtpass,
    ppass,
    NUM_PROGRAMS
};

int timeoutCallback(){
  std::cout << "Timeout callback!" << std::endl;
  // for testing purposes - just ask for abort
  return 1;
} // timeoutCallback

void createContext( Buffer&      points_buffer,
                    Buffer&      temp_buffer,
                    Buffer&      path_buffer,
                    Buffer&      use_buffer)
{
    // Set up context
    context = Context::create();

    context->setRayTypeCount( 2 );
    context->setEntryPointCount( NUM_PROGRAMS ); // chr poner 3u
    context->setStackSize( 2000 );

    context->setTimeoutCallback(timeoutCallback, 20000);

    context["numero_iteraciones"]->setUint( numero_iteraciones );
    context["numero_hilos"]->setUint( numero_hilos );

    context["max_path_length"]->setUint( max_path_length );

    context["scene_epsilon"]->setFloat( scene_epsilon );
    context["beta_diff"]->setFloat( beta_diff );
    context["delta_diff"]->setFloat( delta_diff );
    context["densidad_uniforme"]->setUint( densidad_uniforme );
    context["num_intervalos"]->setUint( num_intervalos );
    context["ancho_intervalo"]->setFloat( ancho_intervalo );
    context["emisividad"]->setFloat( emisividad );
    context["difusion"]->setFloat( difusion );
    
    context["umbral_bbox_hits"]->setUint( umbral_bbox_hits );

    context["bbox_x"]->setFloat( bbox_x );
    context["bbox_y"]->setFloat( bbox_y );
    context["bbox_z"]->setFloat( bbox_z );

    const unsigned int photon_x_intervals = numero_hilos * num_intervalos;

    // POINTS BUFFER
    points_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, numero_hilos );
    points_buffer->setElementSize( sizeof( OriginRecord ) );
    context["ppass_points_buffer"]->set( points_buffer );

    // PATH BUFFER
    path_buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_USER, numero_hilos );
    path_buffer->setElementSize( sizeof( PathRecord ) );
    context["ppass_path_buffer"]->set( path_buffer );

    // TEMP BUFFER
    temp_buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_USER, numero_hilos );
    temp_buffer->setElementSize( sizeof( uint ) );
    context["ppass_temp_buffer"]->set( temp_buffer );

    // USE RECORD
    use_buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_USER, photon_x_intervals );
    use_buffer->setElementSize( sizeof( uint ) );
    context["ppass_use_buffer"]->set( use_buffer );

    // El rtpass se encarga de generar los puntos dentro del solido para procesar con el algoritmo

    {   // rtpass
        const std::string ptx_path = ptxPath( "ppm_rtpass.cu" );
        Program ray_gen_program = context->createProgramFromPTXFile( ptx_path, "rtpass_point_gen" );
        context->setRayGenerationProgram( rtpass, ray_gen_program );
    }

    {   // ppass
        const std::string ptx_path2 = ptxPath( "ppm_ppass.cu");
        Program ray_gen_program2 = context->createProgramFromPTXFile( ptx_path2, "ppass_path_gen" );
        context->setRayGenerationProgram( ppass, ray_gen_program2 );
        context->setMissProgram( ppass, context->createProgramFromPTXFile( ptx_path2, "ppass_miss" ) );
    } // ppass

}

// Utilities for translating Mesh data to OptiX buffers.  These are copied and pasted from sutil.
namespace
{

struct MeshBuffers
{
  optix::Buffer tri_indices;
  optix::Buffer mat_indices;
  optix::Buffer positions;
  optix::Buffer normals;
  optix::Buffer texcoords;
};

void setupMeshLoaderInputs(
    optix::Context            context, 
    MeshBuffers&              buffers,
    Mesh&                     mesh
    )
{
  buffers.tri_indices = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT3,   mesh.num_triangles );
  buffers.mat_indices = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT,    mesh.num_triangles );
  buffers.positions   = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, mesh.num_vertices );
  buffers.normals     = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3,
                                               mesh.has_normals ? mesh.num_vertices : 0);
  buffers.texcoords   = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT2,
                                               mesh.has_texcoords ? mesh.num_vertices : 0);

  mesh.tri_indices = reinterpret_cast<int32_t*>( buffers.tri_indices->map() );
  mesh.mat_indices = reinterpret_cast<int32_t*>( buffers.mat_indices->map() );
  mesh.positions   = reinterpret_cast<float*>  ( buffers.positions->map() );
  mesh.normals     = reinterpret_cast<float*>  ( mesh.has_normals   ? buffers.normals->map()   : 0 );
  mesh.texcoords   = reinterpret_cast<float*>  ( mesh.has_texcoords ? buffers.texcoords->map() : 0 );

  mesh.mat_params = new MaterialParams[ mesh.num_materials ];
}


void unmap( MeshBuffers& buffers, Mesh& mesh )
{
  buffers.tri_indices->unmap();
  buffers.mat_indices->unmap();
  buffers.positions->unmap();
  if( mesh.has_normals )
    buffers.normals->unmap();
  if( mesh.has_texcoords)
    buffers.texcoords->unmap();

  mesh.tri_indices = 0; 
  mesh.mat_indices = 0;
  mesh.positions   = 0;
  mesh.normals     = 0;
  mesh.texcoords   = 0;

  delete [] mesh.mat_params;
  mesh.mat_params = 0;
}

} // namespace

void createGeometry( GeometryGroup& geometry_group ){

    std::string full_path = std::string( sutil::samplesDir() ) + "/assets/" + object_name;

    // We use the base Mesh class rather than OptiXMesh, so we can customize materials below
    // for different passes.
    Mesh mesh;
    MeshLoader loader( full_path );
    loader.scanMesh( mesh );

    MeshBuffers buffers;
    setupMeshLoaderInputs( context, buffers, mesh );

    loader.loadMesh( mesh );

    // Translate to OptiX geometry
    const std::string path = ptxPath( "triangle_mesh.cu" );
    optix::Program bounds_program = context->createProgramFromPTXFile( path, "mesh_bounds" );
    optix::Program intersection_program = context->createProgramFromPTXFile( path, "mesh_intersect" );

    optix::Geometry geometry = context->createGeometry();  
    geometry[ "vertex_buffer"   ]->setBuffer( buffers.positions ); 
    geometry[ "normal_buffer"   ]->setBuffer( buffers.normals); 
    geometry[ "texcoord_buffer" ]->setBuffer( buffers.texcoords ); 
    geometry[ "material_buffer" ]->setBuffer( buffers.mat_indices); 
    geometry[ "index_buffer"    ]->setBuffer( buffers.tri_indices); 
    geometry->setPrimitiveCount     ( mesh.num_triangles );
    geometry->setBoundingBoxProgram ( bounds_program );
    geometry->setIntersectionProgram( intersection_program );

    // Materials have different hit programs depending on pass.
    Program closest_hit  = context->createProgramFromPTXFile( ptxPath( "ppm_ppass.cu" ), "ppass_closest_hit" );
    Program closest_hit2 = context->createProgramFromPTXFile( ptxPath( "ppm_rtpass.cu" ), "rtpass_closest_hit" );

    std::vector< optix::Material > optix_materials;
    for (int i = 0; i < mesh.num_materials; ++i) {

        optix::Material material = context->createMaterial();
        material->setClosestHitProgram( ppass, closest_hit );
        material->setClosestHitProgram( rtpass, closest_hit2 );

        material["Kd"]->set3fv( mesh.mat_params[i].Kd );
        material["Ks"]->set3fv( mesh.mat_params[i].Ks );
        material[ "grid_color" ]->setFloat( 0.5f, 0.5f, 0.5f );
        material[ "use_grid" ]->setUint( mesh.mat_params[i].name == "01_-_Default" ? 1u : 0 );

        optix_materials.push_back( material );
    }

    optix::GeometryInstance geom_instance = context->createGeometryInstance(
            geometry,
            optix_materials.begin(),
            optix_materials.end()
            );

    unmap( buffers, mesh );

    geometry_group->addChild( geom_instance );
    geometry_group->setAcceleration( context->createAcceleration( "Trbvh" ) );

} // createGeometry

void createBbox( GeometryGroup& geometry_group ){
    std::string full_path = std::string( sutil::samplesDir() ) + "/assets/" + bbox_name;

    // We use the base Mesh class rather than OptiXMesh, so we can customize materials below
    // for different passes.
    Mesh mesh;
    MeshLoader loader( full_path );
    loader.scanMesh( mesh );

    MeshBuffers buffers;
    setupMeshLoaderInputs( context, buffers, mesh );

    loader.loadMesh( mesh );

    // Translate to OptiX geometry
    const std::string path = ptxPath( "triangle_mesh.cu" );
    optix::Program bounds_program = context->createProgramFromPTXFile( path, "mesh_bounds" );
    optix::Program intersection_program = context->createProgramFromPTXFile( path, "mesh_intersect" );

    optix::Geometry geometry = context->createGeometry();  
    geometry[ "vertex_buffer"   ]->setBuffer( buffers.positions ); 
    geometry[ "normal_buffer"   ]->setBuffer( buffers.normals); 
    geometry[ "texcoord_buffer" ]->setBuffer( buffers.texcoords ); 
    geometry[ "material_buffer" ]->setBuffer( buffers.mat_indices); 
    geometry[ "index_buffer"    ]->setBuffer( buffers.tri_indices); 
    geometry->setPrimitiveCount     ( mesh.num_triangles );
    geometry->setBoundingBoxProgram ( bounds_program );
    geometry->setIntersectionProgram( intersection_program );

    // Materials have different hit programs depending on pass.
    Program closest_hit = context->createProgramFromPTXFile( ptxPath( "bbox.cu" ), "bbox_closest_hit" );

    std::vector< optix::Material > optix_materials;
    for (int i = 0; i < mesh.num_materials; ++i) {
        optix::Material material = context->createMaterial();
        material->setClosestHitProgram( ppass, closest_hit );
        material->setClosestHitProgram( rtpass, closest_hit );

        material["Kd"]->set3fv( mesh.mat_params[i].Kd );
        material["Ks"]->set3fv( mesh.mat_params[i].Ks );
        material[ "grid_color" ]->setFloat( 0.5f, 0.5f, 0.5f );
        material[ "chr_color" ]->setFloat( 6666.f, 6666.f, 6666.f );
        material[ "use_grid" ]->setUint( mesh.mat_params[i].name == "01_-_Default" ? 1u : 0 );

        optix_materials.push_back( material );
    }

    optix::GeometryInstance geom_instance = context->createGeometryInstance(
            geometry,
            optix_materials.begin(),
            optix_materials.end()
            );

    unmap( buffers, mesh );

    geometry_group->addChild( geom_instance );
    geometry_group->setAcceleration( context->createAcceleration( "Trbvh" ) );

} // createBbox

//------------------------------------------------------------------------------
//
//  Data management
//
//------------------------------------------------------------------------------

 void acumularTemperaturas( uint cant_hilos,
                            uint intervalo_actual,
                            Buffer temp_buffer,
                            Buffer path_buffer,
                            Buffer use_buffer) {

  uint* table_data = reinterpret_cast<uint*>( temp_buffer->map() );

  uint* table_use = reinterpret_cast<uint*>( use_buffer->map() );

  for (uint j = 0; j < cant_hilos; j++){
    if(table_data[j] != Tninguna) {
      // camino con temperatura
      if(table_data[j] == Tmin) {
        // acumular inicios
        CYTmin_array_path[intervalo_actual] += 1;
        // if (! cyril) 
        // acumular resto del camino
        for (uint i = 0; i < num_intervalos; i++){
          if(table_use[j*num_intervalos + i] == usado) {
            // plano usado, acumular pasaje
            Tmin_array_path[i] += 1;
          }
        }
      }
      else if(table_data[j] == Tmax){
        // acumular inicios
        CYTmax_array_path[intervalo_actual] += 1;
        // if (! cyril) 
        // acumular resto del camino
        for (uint i = 0; i < num_intervalos; i++){
          if(table_use[j*num_intervalos + i] == usado) {
            // plano usado, acumular pasaje
            Tmax_array_path[i] += 1;
          }
        }
      }      
    }
  } // for j
  
  caminos_generados += CYTmin_array_path[intervalo_actual] + CYTmax_array_path[intervalo_actual];

  PathRecord* path_data = reinterpret_cast<PathRecord*>( path_buffer->map() );

  for (uint j = 0; j < cant_hilos; j++){
    puntos_generados += path_data[j].puntos_generados;
    caminos_cortados += path_data[j].caminos_cortados;
  } // for

  temp_buffer->unmap();
  use_buffer->unmap();
  path_buffer->unmap();

} // acumularTemperaturas


int organizarArreglo( Buffer points_buffer ){

  OriginRecord* points = reinterpret_cast<OriginRecord*>(points_buffer->map());
// MARCE:
  int cant_digits  = floor( log2(num_cubes)) + 1;
  float step       = precision / num_cubes;
  float offset     = precision / 2;  // para tener todos naturales
  float step_dir   = precision_dir / num_cubes;
  float offset_dir = precision_dir / 2;  // para tener todos naturales
  int   x,y,b;
  int   u,v,w;  
  int   cant_validos = 0;

  for(uint i=0; i < numero_hilos; i++){
    if (points[i].valido){                 // si el punto es válido avanzo
      if (cant_validos != i){              // si los indices son diferentes, hago swap y marco como invalido
    		points[cant_validos].origen = points[i].origen;
    		points[cant_validos].direction = points[i].direction;
    		points[cant_validos].valido = true;
    		// limpio registro invalido
		    points[i].origen            = make_float3(0.f);
    		points[i].direction         = make_float3(0.f);
    		points[i].valido            = false;
      } // if

      // ZORDER
      if (use_zorder){
        x = floor((points[cant_validos].origen.x + offset)/step); // chr
        y = floor((points[cant_validos].origen.y + offset)/step);
        u = floor((points[cant_validos].direction.x + offset_dir)/step_dir);
        v = floor((points[cant_validos].direction.y + offset_dir)/step_dir);
        w = floor((points[cant_validos].direction.z + offset_dir)/step_dir);
        for(int b=0; b< cant_digits; b++) {
          points[cant_validos].zorder |= ((x & (1 << b)) << (b+1) ) | ((y & (1 << b)) << b );    
          // restamos uno para descontar el shifteos
        points[cant_validos].zorder |= ((x & (1 << b)) << ((b*(cant_dimensiones-1))+4) ) \
                                    |  ((y & (1 << b)) << ((b*(cant_dimensiones-1))+3) ) \
                                    |  ((u & (1 << b)) << ((b*(cant_dimensiones-1))+2) ) \
                                    |  ((v & (1 << b)) << ((b*(cant_dimensiones-1))+1) ) \
                                    |  ((w & (1 << b)) << ( b*(cant_dimensiones-1)   ) );      

        } // for
      } // if use_zorder

	    cant_validos++;
    } // if
  } // for


  if (use_zorder){
    std::sort(points, points + cant_validos);
  } // use_zorder


  points_buffer->unmap();

  return cant_validos;

} // organizarArreglo

//------------------------------------------------------------------------------
//
// LAUNCH setup and run 
//
//------------------------------------------------------------------------------

void launch_all(unsigned int numero_hilos,
                uint         plano,
                Buffer       points_buffer,
                Buffer       temp_buffer,
                Buffer       path_buffer,
                Buffer       use_buffer){

    // Hay que declarar un buffer de puntos para que los llene el rtpass
    // El tamaño del buffer es igual a la cantidad de hilos que usamos,
    // así cada hilo tiene un punto para procesar

    int seed = rand();
    context["seed_number"]->setFloat( static_cast<float>( seed ) );

    // Trace photons

    context->launch( rtpass, numero_hilos );
    std::cerr << ":";

    int puntos_validos = 0;

    // organizar arreglo
    puntos_validos = organizarArreglo( points_buffer );
    std::cerr << puntos_validos << ":";

    if (puntos_validos > 0){

      seed = rand();
      context["seed_number"]->setFloat( static_cast<float>( seed ) );

      // Trace photons
      std::cerr << "k<";
      context->launch( ppass, puntos_validos );
      std::cerr << ">, ";

      acumularTemperaturas(puntos_validos, plano, temp_buffer, path_buffer, use_buffer );

    } // if puntos_validos > 0

} // launch_all

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0 ){
    std::cerr << "\nUsage: " << argv0 << " [options]\n";
    std::cerr <<
        "App Options:\n"
        "  -h     | --help                   Print this usage message and exit.\n"
        "  -pl    | --path-length <n>        Largo máximo de los caminos generados. Default = " << max_path_length << ".\n"
        "  -i     | --numero-iteraciones <n> Cantidad de pasadas a realizar sobre la geometría. Default = " << numero_iteraciones << ".\n"
        "  -nh    | --numero-hilos <n>       Cantidad de hilos usados para realizar los cálculos. Default = " << numero_hilos << ".\n"
        "  -p     | --planos <n>             Número de planos en los que se subdivide el modelo. Default = " << num_intervalos << ".\n"
        "  -db    | --delta-b <n>            Valor de delta_b usado para realizar los saltos dentro del sólido. Default = " << beta_diff << ".\n"
        "  -tmin  | --tmin <n>               Temperatura del primer emisor. Default = " << Tmin << ".\n"
        "  -tmax  | --tmax <n>               Temperatura del segundo emisor. Default = " << Tmax << ".\n"
        "  -e     | --emisividad <n>         Emisividad del sólido. Default = " << emisividad << ".\n"
        "  -l     | --lambda <n>             Conductividad térmica del sólido. Default = " << lambda << ".\n"
        "  -du    | --densidad-uniforme      Habilita la densidad uniforme. Default = " << densidad_uniforme << ".\n"
        "  -z     | --use-zorder             Habilita el uso de Z-order. Default = " << use_zorder << ".\n"
        "  -on    | --object-name <n>        Nombre del archivo del modelo a cargar. Default = " << object_name << ".\n"
        "\n"
        << std::endl;

    exit(1);
}


int main( int argc, char** argv ){

    std::string file_name = "nube.txt";
    
    for( int i=1; i<argc; ++i ){

        const std::string arg( argv[i] );

        if( arg == "-h" || arg == "--help" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "-pl" || arg == "--path-length" )
        {
            if( i == argc-1 ){
              std::cerr << "Option '" << arg << "' requires additional argument.\n";
              printUsageAndExit( argv[0] );
            }
            int tmp = atoi( argv[++i] );
            if (tmp > 0) max_path_length = static_cast<unsigned int>(tmp);
        }
        else if( arg == "-i" || arg == "--numero-iteraciones" )
        {
            if( i == argc-1 ){
              std::cerr << "Option '" << arg << "' requires additional argument.\n";
              printUsageAndExit( argv[0] );
            }
            int tmp = atoi( argv[++i] );
            if (tmp > 0) numero_iteraciones = static_cast<unsigned int>(tmp);
        }
        else if( arg == "-nh" || arg == "--numero-hilos" )
        {
            if( i == argc-1 ){
              std::cerr << "Option '" << arg << "' requires additional argument.\n";
              printUsageAndExit( argv[0] );
            }
            int tmp = atoi( argv[++i] );
            if (tmp > 0) numero_hilos = static_cast<unsigned int>(tmp);
        }
        else if( arg == "-p" || arg == "--planos" )
        {
            if( i == argc-1 ){
              std::cerr << "Option '" << arg << "' requires additional argument.\n";
              printUsageAndExit( argv[0] );
            }
            int tmp = atoi( argv[++i] );
            if (tmp > 0) num_intervalos = static_cast<unsigned int>(tmp);
        }
        else if( arg == "-db" || arg == "--delta-b" )
        {
            if( i == argc-1 ){
              std::cerr << "Option '" << arg << "' requires additional argument.\n";
              printUsageAndExit( argv[0] );
            }
            float tmp = atof( argv[++i] );
            if (tmp > 0){
              beta_diff  = static_cast<float>(tmp);
              delta_diff = beta_diff * 0.5f;
            }
        }
        else if( arg == "-tmin" || arg == "--tmin" )
        {
            if( i == argc-1 ){
              std::cerr << "Option '" << arg << "' requires additional argument.\n";
              printUsageAndExit( argv[0] );
            }
            float tmp = atof( argv[++i] );
            if (tmp > 0) valorTmin = static_cast<float>(tmp);
        }
        else if(arg == "-tmax" || arg == "--tmax" )
        {
            if( i == argc-1 ){
              std::cerr << "Option '" << arg << "' requires additional argument.\n";
              printUsageAndExit( argv[0] );
            }
            float tmp = atof( argv[++i] );
            if (tmp > 0) valorTmax = static_cast<float>(tmp);
        }
        else if( arg == "-e" || arg == "--emisividad" )
        {
            if( i == argc-1 ){
              std::cerr << "Option '" << arg << "' requires additional argument.\n";
              printUsageAndExit( argv[0] );
            }
            float tmp = atof( argv[++i] );
            if (tmp > 0) emisividad = static_cast<float>(tmp);
        }
        else if( arg == "-l" || arg == "--lambda" )
        {
            if( i == argc-1 ){
              std::cerr << "Option '" << arg << "' requires additional argument.\n";
              printUsageAndExit( argv[0] );
            }
            float tmp = atof( argv[++i] );
            if (tmp > 0) lambda = static_cast<float>(tmp);
        }
        else if( arg == "-du" || arg == "--densidad-uniforme" )
        {
            densidad_uniforme = true;
        }
        else if( arg == "-z" || arg == "--use-zorder" )
        {
            use_zorder = true;
        }
        else if( arg == "-on" || arg == "--object-name" )
        {
            if( i == argc-1 ){
              std::cerr << "Option '" << arg << "' requires additional argument.\n";
              printUsageAndExit( argv[0] );
            }
            object_name = argv[++i];
        }

        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
        
    } // for argc

    // calculo la difusión
    const float sigma = 5.670373e-8f; //Constante de Stefan-Boltzmann
    float Tref         =(valorTmin + valorTmax) / 2.f;
    float h_r          = 4.f * emisividad * sigma * powf(Tref, 3);
    float cociente_l_b = (lambda / (beta_diff * 0.0001f)); // pasamos beta de decenas de mm a mts
    difusion = cociente_l_b / (cociente_l_b + h_r);

    printf("Tref: %f\nbeta_diff: %f\n",Tref,beta_diff);
    printf("lambda: %f \nhr: %f\ncociente_l_b: %f\n\n",lambda,h_r,cociente_l_b);
    printf("difusion: %f\n", difusion);

    // num_intervalos = (uint)(bbox_z/ancho_intervalo);
    ancho_intervalo = bbox_z/(float)num_intervalos;

    Tmin_array_path   = new uint [num_intervalos];
    Tmax_array_path   = new uint [num_intervalos];
    CYTmin_array_path = new uint [num_intervalos];
    CYTmax_array_path = new uint [num_intervalos];

    for (uint i = 0; i < num_intervalos; ++i){
      Tmin_array_path[i] = 0;
      Tmax_array_path[i] = 0;
      CYTmin_array_path[i] = 0;
      CYTmax_array_path[i] = 0;
    }

    file_name = getFileName("");

    try {

        Buffer points_buffer;
        Buffer temp_buffer;
        Buffer path_buffer;
        Buffer use_buffer;
        createContext( points_buffer, temp_buffer, path_buffer, use_buffer);

        GeometryGroup geometry_group = context->createGeometryGroup();
        createGeometry(geometry_group);
        createBbox(geometry_group);
        context["top_object"]->set( geometry_group );
        context["top_shadower"]->set( geometry_group );

        context->validate();

        // https://devblogs.nvidia.com/how-query-device-properties-and-handle-errors-cuda-cc/
        // int nDevices;
        // cudaGetDeviceCount(&nDevices);
        // std::cerr << "Device count: " << nDevices << std::endl;

        // https://stackoverflow.com/questions/32530604/how-can-i-get-number-of-cores-in-cuda-device
        // cudaDeviceProp prop;
        // cudaGetDeviceProperties(&prop, 0);
        // std::cerr << "getSPcores: " << getSPcores(prop) << std::endl;

        std::cerr << getFileName("") << std::endl;

        double t0_main = sutil::currentTime();

        srand (time(NULL));
        std::string random_string = std::to_string(rand());

        std::string tabla_file_name = getFileName(random_string);
        std::cerr << "tabla_file_name: " << tabla_file_name << std::endl;
        std::cerr << "num_intervalos: "  << num_intervalos << std::endl;
        std::cerr << "ancho_intervalo: " << ancho_intervalo << std::endl;
        std::cerr << "numero_hilos: "    << numero_hilos << std::endl;

        uint iteracion = 0;
        for (iteracion; iteracion < (numero_iteraciones * num_intervalos); iteracion++ ){
          std::cerr << iteracion;
          uint plano = iteracion % num_intervalos;
          context["iteracion"]->setUint( static_cast<uint>( iteracion ) );
          context["plane_number"]->setUint( static_cast<uint>( plano ) );
          launch_all(numero_hilos, plano, points_buffer, temp_buffer, path_buffer, use_buffer);
        } // for

        porcentaje_caminos_cortados = ( (float)caminos_cortados * 100.f) / (caminos_cortados + caminos_generados);

        double time_main = sutil::currentTime() - t0_main;
        std::cerr << "Main while time: " << floor(time_main/60) << "m " << floor(fmod(time_main, 60)) << "s" << std::endl;

        printf("caminos_generados: %u\n", caminos_generados);

        printInfo();

        std::ofstream tabla_out, cyril_tabla_out;
        tabla_out.open(tabla_file_name);
        cyril_tabla_out.open(getFileNameCyril(random_string));

        tabla_out
        << "# numero_iteraciones:  " << numero_iteraciones << "\n"
        << "# numero_hilos:        " << numero_hilos << "\n" 
        << "# max_path_length:     " << max_path_length << "\n" 
        << "# beta_diff:           " << beta_diff << "\n"
        << "# delta_diff:          " << delta_diff << "\n"
        << "# Tmin:                " << valorTmin << "\n"
        << "# Tmax:                " << valorTmax << "\n"
        << "# densidad_uniforme:   " << densidad_uniforme << "\n"
        << "# Cyril:               " << "0" << "\n"
        << "# Z-Order              " << use_zorder << "\n"
        << "# emisividad:          " << emisividad << "\n"
        << "# lambda:              " << lambda << "\n"
        << "# difusion:            " << difusion << "\n"
        << "# num_intervalos:      " << num_intervalos << "\n" 
        << "# umbral_bbox_hits:    " << umbral_bbox_hits << "\n"
        << "# object_name:         " << object_name << "\n"
        << "# caminos_generados:   " << caminos_generados   << "\n"
        << "# caminos_cortados:    " << caminos_cortados   << "\n"
        << "# %% cortados:         " << porcentaje_caminos_cortados   << "\n"
        << "# puntos_generados:    " << puntos_generados << "\n" 
        << "# tiempo:              " << floor(time_main/60) << "m " << floor(fmod(time_main, 60)) << "s" << "\n" 
        << "\n";

        cyril_tabla_out
        << "# numero_iteraciones:  " << numero_iteraciones << "\n"
        << "# numero_hilos:        " << numero_hilos << "\n" 
        << "# max_path_length:     " << max_path_length << "\n" 
        << "# beta_diff:           " << beta_diff << "\n"
        << "# delta_diff:          " << delta_diff << "\n"
        << "# Tmin:                " << valorTmin << "\n"
        << "# Tmax:                " << valorTmax << "\n"
        << "# densidad_uniforme:   " << densidad_uniforme << "\n"
        << "# Cyril:               " << "1" << "\n"
        << "# Z-Order              " << use_zorder << "\n"
        << "# emisividad:          " << emisividad << "\n"
        << "# lambda:              " << lambda << "\n"
        << "# difusion:            " << difusion << "\n"
        << "# num_intervalos:      " << num_intervalos << "\n" 
        << "# umbral_bbox_hits:    " << umbral_bbox_hits << "\n"
        << "# object_name:         " << object_name << "\n"
        << "# caminos_generados:   " << caminos_generados   << "\n"
        << "# caminos_cortados:    " << caminos_cortados   << "\n"
        << "# %% cortados:         " << porcentaje_caminos_cortados   << "\n"
        << "# puntos_generados:    " << puntos_generados << "\n" 
        << "# tiempo:              " << floor(time_main/60) << "m " << floor(fmod(time_main, 60)) << "s" << "\n" 
        << "\n";

        for (uint i = 0; i < num_intervalos; ++i){
          printf("%10d %10d %10d %10d\n", Tmin_array_path[i], Tmax_array_path[i], CYTmin_array_path[i], CYTmax_array_path[i]);
          tabla_out       << std::setw(10) << Tmin_array_path[i]   << " " << std::setw(10) << Tmax_array_path[i]   << std::endl;
          cyril_tabla_out << std::setw(10) << CYTmin_array_path[i] << " " << std::setw(10) << CYTmax_array_path[i] << std::endl;
        }

        tabla_out.close();
        cyril_tabla_out.close();

        destroyContext();
        
        delete Tmin_array_path;
        delete Tmax_array_path;
        delete CYTmin_array_path;
        delete CYTmax_array_path;

        return 0;
    } // try

    SUTIL_CATCH( context->get() )

} // main
