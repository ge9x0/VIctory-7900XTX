//! Local (Wine): Wine's HLSL compiler (vkd3d-shader) leaves a shader's
//! `[RootSignature(RS)]` out of the blob it makes, so `CreateRootSignature`
//! on it fails. This reads the `#define RS "..."` of the source, serializes
//! that signature (version 1.0: everything volatile, the most permissive)
//! and adds it to the shader's container as its `RTS0` part. vkd3d-proton
//! does not check the container's checksum.

use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;

fn tag(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}
fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

/// The parts of a DXBC container: (fourcc, data).
fn parts(c: &[u8]) -> Option<Vec<(u32, &[u8])>> {
    if c.len() < 32 || &c[0..4] != b"DXBC" {
        return None;
    }
    let n = u32_at(c, 28)? as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let at = u32_at(c, 32 + i * 4)? as usize;
        let len = u32_at(c, at + 4)? as usize;
        out.push((u32_at(c, at)?, c.get(at + 8..at + 8 + len)?));
    }
    Some(out)
}

fn container(parts: &[(u32, &[u8])]) -> Vec<u8> {
    let head = 32 + parts.len() * 4;
    let total = head + parts.iter().map(|p| 8 + p.1.len()).sum::<usize>();
    let mut c = Vec::with_capacity(total);
    c.extend_from_slice(b"DXBC");
    c.extend_from_slice(&[0; 16]);
    c.extend_from_slice(&1u32.to_le_bytes());
    c.extend_from_slice(&(total as u32).to_le_bytes());
    c.extend_from_slice(&(parts.len() as u32).to_le_bytes());
    let mut at = head;
    for p in parts {
        c.extend_from_slice(&(at as u32).to_le_bytes());
        at += 8 + p.1.len();
    }
    for p in parts {
        c.extend_from_slice(&p.0.to_le_bytes());
        c.extend_from_slice(&(p.1.len() as u32).to_le_bytes());
        c.extend_from_slice(p.1);
    }
    c
}

/// `code` with the root signature of `source` in it, when it has none and
/// the source defines one; else `code` unchanged.
pub(crate) fn with_root(source: &str, code: Vec<u8>) -> Vec<u8> {
    let rts0 = tag(b"RTS0");
    let Some(have) = parts(&code) else { return code };
    if have.iter().any(|p| p.0 == rts0) {
        return code;
    }
    let Some(text) = source.lines().find_map(|l| l.trim().strip_prefix("#define RS ")).map(|t| t.trim().trim_matches('"').to_owned()) else {
        return code;
    };
    let Some(blob) = serialize(&text) else {
        tracing::warn!(signature = %text, "a root signature of ours not understood (Wine)");
        return code;
    };
    // SAFETY: the blob's own bytes.
    let rs = unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()) };
    let Some(sig) = parts(rs).and_then(|p| p.into_iter().find(|p| p.0 == rts0)) else { return code };
    let mut all = have;
    all.push(sig);
    container(&all)
}

/// Splits at top-level commas.
fn split(s: &str) -> Vec<&str> {
    let (mut d, mut start, mut out) = (0i32, 0, Vec::new());
    for (i, ch) in s.char_indices() {
        match ch {
            '(' => d += 1,
            ')' => d -= 1,
            ',' if d == 0 => {
                out.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    if !s[start..].trim().is_empty() {
        out.push(s[start..].trim());
    }
    out
}

/// `Name(args)` as (name, args).
fn call(s: &str) -> Option<(&str, &str)> {
    let open = s.find('(')?;
    Some((s[..open].trim(), s[open + 1..s.rfind(')')?].trim()))
}

/// A register like `b3` as 3; `key = value` options in a map.
fn reg_and_opts(args: &str) -> (u32, Vec<(String, String)>) {
    let mut reg = 0;
    let mut opts = Vec::new();
    for a in split(args) {
        if let Some((k, v)) = a.split_once('=') {
            opts.push((k.trim().to_ascii_lowercase(), v.trim().to_owned()));
        } else if let Ok(n) = a.trim()[1..].parse() {
            reg = n;
        }
    }
    (reg, opts)
}
fn opt<'a>(o: &'a [(String, String)], k: &str) -> Option<&'a str> {
    o.iter().find(|p| p.0 == k).map(|p| p.1.as_str())
}
fn space(o: &[(String, String)]) -> u32 {
    opt(o, "space").and_then(|v| v.parse().ok()).unwrap_or(0)
}
fn visibility(o: &[(String, String)]) -> D3D12_SHADER_VISIBILITY {
    match opt(o, "visibility").unwrap_or("") {
        "SHADER_VISIBILITY_VERTEX" => D3D12_SHADER_VISIBILITY_VERTEX,
        "SHADER_VISIBILITY_HULL" => D3D12_SHADER_VISIBILITY_HULL,
        "SHADER_VISIBILITY_DOMAIN" => D3D12_SHADER_VISIBILITY_DOMAIN,
        "SHADER_VISIBILITY_GEOMETRY" => D3D12_SHADER_VISIBILITY_GEOMETRY,
        "SHADER_VISIBILITY_PIXEL" => D3D12_SHADER_VISIBILITY_PIXEL,
        _ => D3D12_SHADER_VISIBILITY_ALL,
    }
}
fn address(v: Option<&str>) -> D3D12_TEXTURE_ADDRESS_MODE {
    match v.unwrap_or("TEXTURE_ADDRESS_WRAP") {
        "TEXTURE_ADDRESS_MIRROR" => D3D12_TEXTURE_ADDRESS_MODE_MIRROR,
        "TEXTURE_ADDRESS_CLAMP" => D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        "TEXTURE_ADDRESS_BORDER" => D3D12_TEXTURE_ADDRESS_MODE_BORDER,
        "TEXTURE_ADDRESS_MIRROR_ONCE" => D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE,
        _ => D3D12_TEXTURE_ADDRESS_MODE_WRAP,
    }
}
fn filter(v: Option<&str>) -> Option<D3D12_FILTER> {
    Some(match v.unwrap_or("FILTER_ANISOTROPIC") {
        "FILTER_MIN_MAG_MIP_POINT" => D3D12_FILTER_MIN_MAG_MIP_POINT,
        "FILTER_MIN_MAG_MIP_LINEAR" => D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        "FILTER_MIN_MAG_LINEAR_MIP_POINT" => D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT,
        "FILTER_ANISOTROPIC" => D3D12_FILTER_ANISOTROPIC,
        _ => return None,
    })
}

fn serialize(text: &str) -> Option<ID3DBlob> {
    let mut flags = D3D12_ROOT_SIGNATURE_FLAGS(0);
    let mut params: Vec<D3D12_ROOT_PARAMETER> = Vec::new();
    let mut ranges: Vec<Vec<D3D12_DESCRIPTOR_RANGE>> = Vec::new();
    let mut table_of: Vec<Option<usize>> = Vec::new();
    let mut samplers = Vec::new();
    for item in split(text) {
        let (name, args) = call(item)?;
        match name {
            "RootFlags" => {
                for f in args.split('|').map(str::trim) {
                    flags.0 |= match f {
                        "0" | "" => 0,
                        "ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT" => D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT.0,
                        "CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED" => D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED.0,
                        "SAMPLER_HEAP_DIRECTLY_INDEXED" => D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED.0,
                        _ => return None,
                    };
                }
            }
            "RootConstants" => {
                let (reg, o) = reg_and_opts(args);
                let n = opt(&o, "num32bitconstants")?.parse().ok()?;
                params.push(D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: reg, RegisterSpace: space(&o), Num32BitValues: n } },
                    ShaderVisibility: visibility(&o),
                });
                table_of.push(None);
            }
            "CBV" | "SRV" | "UAV" => {
                let (reg, o) = reg_and_opts(args);
                let kind = match name {
                    "CBV" => D3D12_ROOT_PARAMETER_TYPE_CBV,
                    "SRV" => D3D12_ROOT_PARAMETER_TYPE_SRV,
                    _ => D3D12_ROOT_PARAMETER_TYPE_UAV,
                };
                params.push(D3D12_ROOT_PARAMETER {
                    ParameterType: kind,
                    Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: reg, RegisterSpace: space(&o) } },
                    ShaderVisibility: visibility(&o),
                });
                table_of.push(None);
            }
            "DescriptorTable" => {
                let mut list = Vec::new();
                let mut vis = D3D12_SHADER_VISIBILITY_ALL;
                for part in split(args) {
                    if let Some((k, v)) = part.split_once('=').filter(|_| !part.contains('(')) {
                        if k.trim().eq_ignore_ascii_case("visibility") {
                            vis = visibility(&[("visibility".into(), v.trim().into())]);
                        }
                        continue;
                    }
                    let (kind, a) = call(part)?;
                    let (reg, o) = reg_and_opts(a);
                    let n = match opt(&o, "numdescriptors") {
                        Some("unbounded") => u32::MAX,
                        Some(v) => v.parse().ok()?,
                        None => 1,
                    };
                    list.push(D3D12_DESCRIPTOR_RANGE {
                        RangeType: match kind {
                            "SRV" => D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                            "UAV" => D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                            "CBV" => D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
                            "Sampler" => D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
                            _ => return None,
                        },
                        NumDescriptors: n,
                        BaseShaderRegister: reg,
                        RegisterSpace: space(&o),
                        OffsetInDescriptorsFromTableStart: opt(&o, "offset").and_then(|v| v.parse().ok()).unwrap_or(D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND),
                    });
                }
                ranges.push(list);
                params.push(D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                    Anonymous: D3D12_ROOT_PARAMETER_0 { DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE::default() },
                    ShaderVisibility: vis,
                });
                table_of.push(Some(ranges.len() - 1));
            }
            "StaticSampler" => {
                let (reg, o) = reg_and_opts(args);
                samplers.push(D3D12_STATIC_SAMPLER_DESC {
                    Filter: filter(opt(&o, "filter"))?,
                    AddressU: address(opt(&o, "addressu")),
                    AddressV: address(opt(&o, "addressv")),
                    AddressW: address(opt(&o, "addressw")),
                    MipLODBias: 0.0,
                    MaxAnisotropy: 16,
                    ComparisonFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
                    BorderColor: D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
                    MinLOD: 0.0,
                    MaxLOD: f32::MAX,
                    ShaderRegister: reg,
                    RegisterSpace: space(&o),
                    ShaderVisibility: visibility(&o),
                });
            }
            _ => return None,
        }
    }
    for (p, t) in params.iter_mut().zip(&table_of) {
        if let Some(t) = t {
            p.Anonymous.DescriptorTable = D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: ranges[*t].len() as u32, pDescriptorRanges: ranges[*t].as_ptr() };
        }
    }
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: samplers.len() as u32,
        pStaticSamplers: samplers.as_ptr(),
        Flags: flags,
    };
    let mut blob = None;
    // SAFETY: the description and what it points at live through the call.
    unsafe { D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None).ok()? };
    blob
}
