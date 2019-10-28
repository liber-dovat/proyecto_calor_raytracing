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
#include "helpers.h"
#include "ppm.h"
#include "random.h"

using namespace optix;

//
// Scene wide variables
//
rtDeclareVariable(uint,  umbral_bbox_hits, , );
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(float, bbox_x, , );
rtDeclareVariable(float, bbox_y, , );
rtDeclareVariable(float, bbox_z, , );

//
// Closest hit material
//
rtDeclareVariable(float3,     geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(optix::Ray, ray,              rtCurrentRay, );
rtDeclareVariable(float,      t_hit,            rtIntersectionDistance, );
rtDeclareVariable(PhotonPRD,  hit_record,       rtPayload, );

RT_PROGRAM void bbox_closest_hit(){

  if (hit_record.bbox_hits >= umbral_bbox_hits){
    hit_record.cortar_camino = 8;
    return;
  }

  float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
  float dot_product = dot(world_geometric_normal, ray.direction);

  float3 hit_point = ray.origin + t_hit*ray.direction;

  // si un rayo viene de afuera lo termino
  if (dot_product < 0){
    hit_record.cortar_camino = 3;
    return;
  }

  // choco con fuente
  if (abs(hit_point.z) > (bbox_z*0.5f - scene_epsilon)){ // choco con fuente, no tiro rayo

    // si es el rayo test y choco con tapa(generado dentro del solido) lo teleporto a la otra tapa para que apunte a la geometria.
    if ( hit_record.init_camino && hit_record.ray_type == inf_ray ) {
      // teleportar en z
      hit_point.z          = -hit_point.z;
    } else if (hit_point.z > (bbox_z*0.5f - scene_epsilon)){ // sino asigno las temperaturas
      hit_record.temperature = Tmax;
    } else if (hit_point.z < (-bbox_z*0.5f + scene_epsilon)){
      hit_record.temperature = Tmin;
    } // if 

  }else{

    hit_record.bbox_hits++;

    // hubo un hit con la caja, y estoy dentro, entonces discrimino el teleport del rayo
    // si la coordenada z >= emisor.z retorno temp del emisor
    // la direccion se mantiene, y solo se cambia el origen del rayo

    // cambio coordenadas del rayo
    if (abs(hit_point.y) > (bbox_y*0.5f - scene_epsilon)){

      if (hit_point.y > 0)
         hit_point.y = -hit_point.y + scene_epsilon;
      else
         hit_point.y = -hit_point.y - scene_epsilon;

    } // if y > bbox.y

    if (abs(hit_point.x) > (bbox_x*0.5f - scene_epsilon)){

      if (hit_point.x > 0)
         hit_point.x = -hit_point.x + scene_epsilon;
      else
         hit_point.x = -hit_point.x - scene_epsilon;

    } // if x > bbox.x
  } // if choco con fuente

  hit_record.position   = hit_point;
  hit_record.direction  = ray.direction;
} // bbox_closest_hit
