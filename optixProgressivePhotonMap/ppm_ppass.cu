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

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include <curand.h>
#include <curand_kernel.h>

#include "helpers.h"
#include "ppm.h"
#include "random.h"

using namespace optix;

//
// Scene wide variables
//
rtDeclareVariable(uint,     numero_hilos, , );
rtDeclareVariable(uint,     plane_number , , );
rtDeclareVariable(float,    scene_epsilon, , );
rtDeclareVariable(float,    beta_diff, , );
rtDeclareVariable(float,    delta_diff, , );
rtDeclareVariable(float,    emisividad, , );
rtDeclareVariable(float,    difusion, , );
rtDeclareVariable(uint,     densidad_uniforme, , );
rtDeclareVariable(float,    bbox_x, , );
rtDeclareVariable(float,    bbox_y, , );
rtDeclareVariable(float,    bbox_z, , );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(uint,     iteracion , , );
rtDeclareVariable(float,    seed_number , , );
//
// Ray generation program
//
rtBuffer<OriginRecord, 1>   ppass_points_buffer; // record :Tmin,Tmax
rtBuffer<uint, 1>           ppass_temp_buffer; // Tmin,Tmax,Tninguna
rtBuffer<uint, 1>           ppass_use_buffer; // noUsado,usado
rtBuffer<PathRecord, 1>     ppass_path_buffer; // record :Tmin,Tmax
rtDeclareVariable(uint,     num_intervalos, , );
rtDeclareVariable(float,    ancho_intervalo, , );
rtDeclareVariable(uint,     max_path_length, , );

rtDeclareVariable(uint2,    launch_index, rtLaunchIndex, );

// http://cs.umw.edu/~finlayson/class/fall16/cpsc425/notes/cuda-random.html
// http://richiesams.blogspot.com/2015/03/creating-randomness-and-acummulating.html

/* this GPU kernel function calculates a random number and stores it in the parameter */
static __device__ __inline__ void cuRandom(curandState_t& state, unsigned int& result) {

    /* curand works like rand - except that it takes a state as a parameter */
    result = curand(&state);
} // cuRandom

// regresa en &result un valor random entre [0,1]
static __device__ __inline__ void cuRandom01(curandState_t& state, float& result) {

  unsigned int gpu_x;
  cuRandom(state, gpu_x);
  float nrd_tmp = (float) gpu_x;
  result = fmod(nrd_tmp, 100000000.f) / 100000000.f;

} // cuRandom01

// sample hemisphere with cosine density
static __device__ __inline__ void cosineHemisphere( const optix::float3 sample ,
                                                    const optix::float3& U,
                                                    const optix::float3& V,
                                                    const optix::float3& W,
                                                    optix::float3& point )
{

    float phi = 2.0f * M_PIf*sample.x;
    float r = (float)sqrt( sample.y );
    float x = r * (float)cos(phi);
    float y = r * (float)sin(phi);
    float z = 1.0f - sample.z;
    z = z > 0.0f ? (float)sqrt(z) : 0.0f;

    point = x*U + y*V + z*W;

} // cosineHemisphere

// uniform sample hemisphere
static __device__ __inline__ void uniformHemisphere( const optix::float3 sample,
                                                     const optix::float3& U,
                                                     const optix::float3& V,
                                                     const optix::float3& W,
                                                     optix::float3& point )
{

    float phi   = 2.f * M_PIf * sample.x;
    float theta = acos(1 - sample.y);
    float x     = sin(theta) * cos(phi);
    float y     = sin(theta) * sin(phi);
    float z     = cos(theta);

    point = x*U + y*V + z*W;
} // uniformHemisphere

// http://mathworld.wolfram.com/SpherePointPicking.html
// http://corysimon.github.io/articles/uniformdistn-on-sphere/

// uniform sample sphere
static __device__ __inline__ void uniformSphere( const optix::float3 sample,
                                                 const optix::float3& U,
                                                 const optix::float3& V,
                                                 const optix::float3& W,
                                                 optix::float3& point )
{

    float phi   = 2.f * M_PIf * sample.x;
    float theta = acos(2.f * sample.y - 1);
    float x     = sin(theta) * cos(phi);
    float y     = sin(theta) * sin(phi);
    float z     = cos(theta);

    point = x*U + y*V + z*W;

} // uniformSphere

// http://www.rorydriscoll.com/2009/01/07/better-sampling/

RT_PROGRAM void ppass_path_gen(){

  uint pm_index = launch_index.x * num_intervalos;

  float3 ray_origin, ray_direction, sample;
  float2 sample2;
  float ray_tmax;
  curandState_t state;
  
  if(!ppass_points_buffer[launch_index.x].valido) {
    printf("Punto %d invalido!\n", launch_index.x);  
    return;
  }
  
  curand_init(pm_index+(uint)seed_number+iteracion, 0, 0, &state);

  // inicializo el buffer de informacion extra de los caminos
  ppass_path_buffer[launch_index.x].caminos_generados = 0u;
  ppass_path_buffer[launch_index.x].caminos_cortados  = 0u;
  ppass_path_buffer[launch_index.x].puntos_generados  = 0u;

  // Inicializar los caminos 

  for(unsigned int i = 0; i < num_intervalos; ++i) {
    ppass_use_buffer [pm_index + i] = noUsado;
  } // for
  ppass_temp_buffer[launch_index.x] = Tninguna; 

  ppass_use_buffer[pm_index + plane_number] = usado;
  ppass_path_buffer[launch_index.x].puntos_generados++; // genero un punto    

  ray_origin = ppass_points_buffer[launch_index.x].origen; // con esto obtengo el punto de origen precalculado
  ray_direction = ppass_points_buffer[launch_index.x].direction; // con esto la direccion de origen precalculado

  PhotonPRD prd;
  prd.temperature   = Tninguna;
  prd.trace_tmax    = delta_diff;
  prd.ray_type      = delta_ray;
  prd.position      = ray_origin;
  prd.test_position = ray_origin;
  prd.direction     = ray_direction;
  prd.pm_index      = pm_index;
  prd.num_deposits  = 0;
  prd.bbox_hits     = 0;
  prd.cortar_camino = 0;   // revisar mas adelante por reinicializacion
  
  uint i_safe    = 0;
  uint num_paths = 0;
  
  bool tengo_camino = false;

  while( !tengo_camino ) {

    cuRandom01(state, sample.x);
    cuRandom01(state, sample.y);
    cuRandom01(state, sample.z);
    cuRandom01(state, sample2.x);
    cuRandom01(state, sample2.y);
    prd.sample       = sample;
    prd.rand_reflex  = sample2.x;
    prd.rand_diffuse = sample2.y;

    ray_tmax      = prd.trace_tmax;
    ray_origin    = prd.position;
    ray_direction = prd.direction;

    uint prev_num_deposits = prd.num_deposits;

    optix::Ray ray = make_Ray(ray_origin, ray_direction, ppass_and_gather_ray_type, scene_epsilon, ray_tmax ); 
    rtTrace( top_object, ray, prd );

    if (prev_num_deposits < prd.num_deposits) {
      i_safe=0;
    } // if
    i_safe++; // me aseguro de cortar el while por si num_deposits no incrementa

    if (prd.cortar_camino > 0 || i_safe > max_path_length){
      // Si cortó por return:

      tengo_camino = true;

      ppass_path_buffer[launch_index.x].caminos_cortados++;
      
    } else if (prd.temperature != Tninguna){
      // el último punto del camino llegó a una fuente

      ppass_temp_buffer[launch_index.x] = prd.temperature; 
      
      tengo_camino = true;
      num_paths++;

    } // if

  } // while

  // guardo la cantida de caminos generados
  ppass_path_buffer[launch_index.x].caminos_generados = num_paths;

  if (i_safe >= max_path_length){
    prd.cortar_camino = 7;
  }

} // ppass_path_gen

//
// Closest hit material
//
rtDeclareVariable(float3,     geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3,     shading_normal, attribute shading_normal, ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float,      t_hit, rtIntersectionDistance, );
rtDeclareVariable(PhotonPRD,  hit_record, rtPayload, );

static __device__ __inline__ uint getPosition(float3 point) {

  // el primer operando hace un shift de las posiciones z a valores positivos
  // el segundo es la cantida de grillas calculado como el largo dividido el ancho de los intervalos
  // luego la pos es Z>0 modulo el numero de celdas en la grilla

  float a = point.z + (bbox_z * 0.5f) - scene_epsilon;
  float b = ancho_intervalo;

  uint p = (uint)(a/b);
  
  return p;
}

static __device__ __inline__ bool cortaPlano(float3 origen, float3 destino) {
  // calcular si vector (origen,destino) corta plano de intervalo destino
  uint p_destino = getPosition(destino);
  float plano_z = (p_destino*ancho_intervalo)-(bbox_z*0.5f - scene_epsilon)+(ancho_intervalo/2.0f);
  
  return !(((origen.z - plano_z) * (destino.z - plano_z)) > 0);
}

RT_PROGRAM void ppass_closest_hit(){

  float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 ffnormal               = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

  float3 hit_point = ray.origin + t_hit*ray.direction;
  float3 new_ray_dir;
  float  new_tmax = RT_DEFAULT_MAX;

  hit_record.bbox_hits = 0; // reinicio los golpes con la bbox

  float dot_product = dot(world_geometric_normal, ray.direction);

  // di un hit y no soy el primer rayo. Determinar si viene de dentro o de fuera
  // supongo que estoy adentro

  if (dot_product < 0 && hit_record.ray_type != inf_ray){ // si soy un rayo delta o beta externo (>0) es invalido
    hit_record.cortar_camino = 1; // corto el camino
    return;
  }

  if (dot_product >= 0 && hit_record.ray_type == inf_ray){ // si es un rayo infinito interno es invalido corto el camino
    hit_record.cortar_camino = 4;
    return;
  } // if

  bool usarSuperficie = true;
    
  if (dot_product < 0) { // vengo desde fuera

    usarSuperficie = (hit_record.rand_reflex < emisividad);

    // si no uso la superficie, reboto con coseno hacia el exterior
    if (!usarSuperficie){
      // reflejo hacia afuera
      new_tmax            = RT_DEFAULT_MAX;
      hit_record.ray_type = inf_ray;
      float3 U, V, W;
      create_onb(ffnormal, U, V, W);
      cosineHemisphere(hit_record.sample, U, V, W, new_ray_dir);
    } 
  } // if not inf_ray

  // Si uso la superficie distingo dos casos, soy delta o inf

  if (usarSuperficie) {
    bool ingreso = hit_record.rand_reflex < difusion;
    // calculo nuevo rayo segun salgo o entro

    if (ingreso){
      new_tmax            = beta_diff; // reboto usando la normal del hit_point
      hit_record.ray_type = beta_ray;

      if (dot_product < 0) { // vengo de fuera
        new_ray_dir = -ffnormal;
      }else{ // vengo de dentro
        new_ray_dir = ffnormal;
      }

    }else{
      new_tmax            = RT_DEFAULT_MAX;
      hit_record.ray_type = inf_ray;
      float3 U, V, W;

      if (dot_product < 0) { // vengo de fuera
        ffnormal = ffnormal;
      }else{ // vengo de dentro
        ffnormal = -ffnormal;
      }

      create_onb(ffnormal, U, V, W);
      cosineHemisphere(hit_record.sample, U, V, W, new_ray_dir);
    } // if ingreso

  } //if usarSuperficie

  // guardo info del choque
  if(cortaPlano(ray.origin, hit_point) && (dot_product >= 0)) {
    ppass_use_buffer[hit_record.pm_index + getPosition(hit_point)] = usado;
    ppass_path_buffer[launch_index.x].puntos_generados++; // genero un punto
  }

  hit_record.num_deposits++;

  // si me paso del numero de fotones del camino reinicio camino
  if ( hit_record.num_deposits >= max_path_length ){
    hit_record.cortar_camino = 6;
    return;
  }

  hit_record.position   = hit_point;
  hit_record.direction  = new_ray_dir;
  hit_record.trace_tmax = new_tmax;

} // ppass_closest_hit2

// miss
// si ray.tmax == INIFNITY retorno
// sino 
// estoy adentro, guardo punto, calculo otra direccion y tiro otro rayo con tmax=delta_diff.

RT_PROGRAM void ppass_miss(){

  // si el rayo se iba a infinito, es decir, salía del objeto y dio miss con la escena, termino y retorno
  if (hit_record.ray_type == inf_ray){
    hit_record.cortar_camino = 2;
    return;
  } else if (hit_record.ray_type == delta_ray || hit_record.ray_type == beta_ray){ // si venía con vector delta_diff y dio miss estoy adentro

    // calculo el hit point como el origen mas delta/beta segun energy.z
    float tmaxTemp = inf_ray;
    if (hit_record.ray_type == delta_ray ){
      tmaxTemp = delta_diff;
    }else if (hit_record.ray_type == beta_ray ){
      tmaxTemp = beta_diff;
    } // if
        
    float3 new_coord = ray.origin + (tmaxTemp - scene_epsilon) * ray.direction;
    float3 new_ray_dir;

    float3 U, V, W;
    create_onb(ray.direction, U, V, W);
    uniformSphere(hit_record.sample, U, V, W, new_ray_dir);

    // guardo punto
    if(cortaPlano(ray.origin, new_coord)) {
      ppass_use_buffer[hit_record.pm_index + getPosition(new_coord)] = usado;
      ppass_path_buffer[launch_index.x].puntos_generados++; // genero un punto      
    }

    hit_record.ray_type = delta_ray;
    hit_record.num_deposits++;

    // si me paso del numero de fotones del camino reinicio camino
    if ( hit_record.num_deposits >= max_path_length){
      hit_record.cortar_camino = 6;
      return;
    }

    hit_record.position   = new_coord;
    hit_record.direction  = new_ray_dir;
    hit_record.trace_tmax = delta_diff;

  } // if ray.tmax

} // rtpass_miss2
