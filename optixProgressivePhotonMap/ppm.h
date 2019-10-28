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

#include <optixu/optixu_math_namespace.h>

// tipos de temperatura
#define  Tmin     0
#define  Tmax     1
#define  Tninguna 2

// tipos de uso
#define noUsado 0
#define usado   1

enum RayTypes
{
    rtpass_ray_type,
    ppass_and_gather_ray_type
};

enum RayLength
{
    inf_ray,
    delta_ray,
    beta_ray
};

struct OriginRecord
{
	optix::float3 origen;        // 12 bytes
	optix::float3 direction;        // 12 bytes
	bool          valido;        // 1  bytes
  optix::uint   zorder;        // 4 bytes

  OriginRecord(optix::float3& o, bool& v, optix::uint& z) : origen(o), valido(v), zorder(z) {}

  bool operator < (const OriginRecord& elem) const {
      return (zorder < elem.zorder);
  } // operator <

};

struct PathRecord
{
  optix::uint caminos_generados;       // 4 bytes
  optix::uint caminos_cortados;        // 4 bytes
  optix::uint puntos_generados;        // 4 bytes
};

// bool, char, unsigned char, signed char, __int8         1 byte
// __int16, short, unsigned short, wchar_t, __wchar_t     2 bytes
// float, __int32, int, unsigned int, long, unsigned long 4 bytes
// double, __int64, long double, long long                8 bytes

struct PhotonRecord          // 61 bytes
{
  optix::float3 position;    // 12 bytes
  optix::float3 normal;      // 12 bytes
  optix::float3 ray_dir;     // 12 bytes
  float         temperature; // 4 bytes
  float         trace_tmax;  // 4 bytes
  float         ray_type;    // 4 bytes
  optix::uint   axis;        // 4 bytes
  optix::uint   bbox_hits;   // 4 bytes
  bool          init_camino; // 1 byte
  // -- extra bytes
  optix::uint   cortar_camino; // 4 bytes
};

struct PhotonPRD
{
  optix::float3 position;
  optix::float3 test_position;
  optix::float3 direction;
  optix::uint   temperature;
  float         trace_tmax;
  float         ray_type;
  optix::uint   pm_index;
  optix::float3 sample;
  float         rand_reflex;
  float         rand_diffuse;
  optix::uint   num_deposits;
  optix::uint   bbox_hits;   // 4 bytes
  bool          init_camino;

  optix::uint   cortar_camino; 
  // 1 doy hit vengo de fuera y soy delta,
  // 2 miss con rayo infinito, 
  // 3 doy hit con bbox desde fuera, 
  // 4 doy hit vengo de adentro y soy infinito, 
  // 5 vengo de afuera y soy test
  // 6 max photon count
  // 7 salgo por isafe
  // 8 max teleports
};
