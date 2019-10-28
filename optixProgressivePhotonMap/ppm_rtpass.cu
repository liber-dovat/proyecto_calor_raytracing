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

rtDeclareVariable(uint,     numero_hilos, , );
rtDeclareVariable(float,    scene_epsilon, , );
rtDeclareVariable(float,    ancho_intervalo, , );
rtDeclareVariable(uint,     densidad_uniforme, , );
rtDeclareVariable(uint,     umbral_bbox_hits, , );
rtDeclareVariable(float,    bbox_x, , );
rtDeclareVariable(float,    bbox_y, , );
rtDeclareVariable(float,    bbox_z, , );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(uint,     plane_number , , );
rtDeclareVariable(uint,     iteracion , , );
rtDeclareVariable(float,    seed_number , , );

rtBuffer<OriginRecord, 1>   ppass_points_buffer;
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

// https://devtalk.nvidia.com/default/topic/825579/optix/manipulating-geometry-with-keyboard-or-mouse/
// https://devtalk.nvidia.com/default/topic/853732/optix/-solved-triaid-in-optix-and-other/
/*
static __device__ __inline__ void getBboxPhoton(curandState_t& state, float3& o, float3& d){
  
  float sample_x, sample_y, sample_z;
  cuRandom01(state, sample_x);
  cuRandom01(state, sample_y);
  cuRandom01(state, sample_z);
  float3 sample;
  cuRandom01(state, sample.x);
  cuRandom01(state, sample.y);
  cuRandom01(state, sample.z);
  
  // genero un nuevo vector al azar en su misma recta y no mayor en modulo
  float new_x = (bbox_x*0.5f - scene_epsilon) * (2.f*sample_x - 1.f);
  float new_y = (bbox_y*0.5f - scene_epsilon) * (2.f*sample_y - 1.f);
  float new_z = (bbox_z*0.5f - scene_epsilon) * (2.f*sample_z - 1.f);

  o = make_float3(new_x, new_y, new_z);
  float3 normal = make_float3(1.f, 0.f, 0.f);

  float3 U, V, W;
  create_onb(normal, U, V, W);
  uniformSphere(sample, U, V, W, d);
  
} // getBboxPhoton
*/

static __device__ __inline__ void getPlanePhoton(curandState_t& state, uint intervalo, float3& o, float3& d){
  
  float sample_x, sample_y;
  cuRandom01(state, sample_x);
  cuRandom01(state, sample_y);
  float3 sample;
  cuRandom01(state, sample.x);
  cuRandom01(state, sample.y);
  cuRandom01(state, sample.z);

  // genero un nuevo vector al azar en su misma recta y no mayor en modulo
  float new_x = (bbox_x*0.5f - scene_epsilon) * (2.f*sample_x - 1.f);
  float new_y = (bbox_y*0.5f - scene_epsilon) * (2.f*sample_y - 1.f);
  float new_z = (intervalo*ancho_intervalo)-(bbox_z*0.5f - scene_epsilon)+(ancho_intervalo/2.0f);

  o = make_float3(new_x, new_y, new_z);
  float3 normal = make_float3(1.f, 0.f, 0.f);

  float3 U, V, W;
  create_onb(normal, U, V, W);
  uniformSphere(sample, U, V, W, d);
  
} // getBboxPhoton

RT_PROGRAM void rtpass_point_gen(){

  // Cada hilo se encarga de conseguir un punto dentro del sólido y colocar su coordenada en el buffer

  ppass_points_buffer[launch_index.x].origen = make_float3(0.f); // inicializo la celda que voy a calcular
  ppass_points_buffer[launch_index.x].valido = false;
  ppass_points_buffer[launch_index.x].zorder = 0u;

  float3 ray_origin, ray_direction;

  PhotonPRD prd;
  prd.temperature   = Tninguna;
  prd.trace_tmax    = RT_DEFAULT_MAX;
  prd.ray_type      = inf_ray;
  prd.pm_index      = launch_index.x;
  prd.num_deposits  = 0;
  prd.bbox_hits     = 0;
  prd.cortar_camino = 0;   // revisar mas adelante por reinicializacion
  prd.init_camino   = true;

  bool tengo_punto = false;

  curandState_t state;
  curand_init(launch_index.x+(uint)seed_number+iteracion, 0, 0, &state);

  getPlanePhoton(state, plane_number, ray_origin, ray_direction);

  prd.position      = ray_origin;
  prd.test_position = ray_origin;
  prd.direction     = ray_direction;

  uint max_intentos = 25 * umbral_bbox_hits;

  int intentos = 0;
  while (!tengo_punto && (intentos < max_intentos)){

    // dado el plano actual que vamos a procesar obtengo un punto al azar en el, y luego hago un trace para saber
    // si estoy dentro o fuera del sólido.

    ray_origin    = prd.position;
    ray_direction = prd.direction;

    optix::Ray ray = make_Ray(ray_origin, ray_direction, rtpass_ray_type, scene_epsilon, RT_DEFAULT_MAX ); 
    rtTrace( top_object, ray, prd );

    if (!prd.init_camino){ // estoy adentro
      
      tengo_punto = true;
	    ppass_points_buffer[launch_index.x].valido    = true;
	    ppass_points_buffer[launch_index.x].origen    = prd.test_position; // asigno el punto de comienzo al buffer
	    ppass_points_buffer[launch_index.x].direction = prd.direction; // asigno la dir del punto de comienzo al buffer
    
    } else if (prd.cortar_camino > 0){ // si hay error

      if (densidad_uniforme && (prd.cortar_camino == 5)){ // estoy afuera

        tengo_punto = true;
		    ppass_points_buffer[launch_index.x].valido = false;

      }else{ // si tengo que obtener un punto si o si, calculo una nueva posición
 
        getPlanePhoton(state, plane_number, ray_origin, ray_direction); 

        prd.temperature   = Tninguna;
        prd.trace_tmax    = RT_DEFAULT_MAX;
        prd.ray_type      = inf_ray;
        prd.pm_index      = launch_index.x;
        prd.num_deposits  = 0;
        prd.bbox_hits     = 0;
        prd.cortar_camino = 0;   // reinicializacion
        prd.init_camino   = true;
        prd.position      = ray_origin;
        prd.test_position = ray_origin;
        prd.direction     = ray_direction;
      }

    } // if

    intentos++;

  } // while

} // rtpass_point_gen

rtDeclareVariable(float3,     geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3,     shading_normal, attribute shading_normal, ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(PhotonPRD,  hit_record, rtPayload, );

RT_PROGRAM void rtpass_closest_hit(){

  float3 world_shading_normal   = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, shading_normal ) );
  float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float3 ffnormal               = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

  float dot_product = dot(world_geometric_normal, ray.direction);

  if (dot_product > 0){ // soy un punto interno del sólido
    hit_record.init_camino = false;
  }else{ // vengo de fuera y soy un rayo test
    hit_record.cortar_camino = 5;
  } // if

} // rtpass_closest_hit
