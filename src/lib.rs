use anyhow::Result;
#[cfg(feature = "shader-structs")]
pub use spirv_reflect::types::image::ReflectFormat;
#[cfg(feature = "shader-structs")]
use spirv_reflect::types::{ReflectDecorationFlags, ReflectTypeFlags};
use spirv_reflect::ShaderModule;
use std::{
    fs,
    path::{Path, PathBuf},
};
#[cfg(feature = "shader-structs")]
use syn::parse_quote;
use thiserror::Error;
pub use {
    shaderc::{EnvVersion, OptimizationLevel},
    spirv_reflect::types::{
        descriptor::ReflectDescriptorType, op::ReflectOp, variable::ReflectShaderStageFlags,
        ReflectBlockVariable, ReflectEntryPointLocalSize,
    },
};

#[derive(Debug, Error)]
pub enum ShaderCompilationError {
    #[error("could not create shader compiler")]
    CompilerCreationFailed,
    #[error("could not create shader compiler options")]
    CompilerOptionsCreationFailed,
    #[error("could not get shader output path")]
    CouldNotGetShaderOutputPath,
    #[error("the shader path is not a valid UTF-8 string")]
    NonUtf8Path(std::path::PathBuf),
    #[error("path does not contain a valid file name")]
    NoFileName(std::path::PathBuf),
    #[error("Could not compiler shader: {0}")]
    CompilationFailed(shaderc::Error),
    #[error("Could not create directory: {0}")]
    CouldNotCreateDir(std::io::Error),
}

// TODO: This is really bad, need to figure out a much better way to handle this
fn is_low_precision(name: &str) -> bool {
    name.contains("_lowp")
}

pub fn is_runtime_array(op: ReflectOp) -> bool {
    *op == spirv::Op::TypeRuntimeArray
}

#[cfg(feature = "shader-structs")]
fn field_from_ident_and_type(ident: syn::Ident, ty: syn::Type) -> syn::Field {
    syn::Field {
        attrs: vec![],
        vis: syn::Visibility::Public(syn::VisPublic {
            pub_token: syn::token::Pub::default(),
        }),
        ident: Some(ident),
        colon_token: Some(Default::default()),
        ty,
    }
}

#[cfg(feature = "shader-structs")]
fn struct_from_fields(struct_name: &str, fields: &[syn::Field], archive: bool) -> syn::ItemStruct {
    let struct_ident = syn::Ident::new(struct_name, proc_macro2::Span::call_site());
    if archive {
        parse_quote! {
            #[repr(C)]
            #[derive(Debug, Default, Copy, Clone)]
            #[derive(rkyv::Archive, rkyv::Deserialize, rkyv::Serialize)]
            pub struct #struct_ident {
                #(#fields,)*
            }
        }
    } else {
        parse_quote! {
            #[repr(C)]
            #[derive(Debug, Default, Copy, Clone)]
            pub struct #struct_ident {
                #(#fields,)*
            }
        }
    }
}

// TODO: This is largely very bad
#[cfg(feature = "shader-structs")]
pub fn get_structs_from_blocks(blocks: &[ReflectBlockVariable]) -> Vec<ShaderStruct> {
    let ty_descriptors = blocks
        .iter()
        .filter(|block| block.type_description.is_some())
        .map(|block| block.type_description.as_ref().unwrap().clone())
        .collect::<Vec<_>>();
    let structs = ty_descriptors
        .iter()
        .filter(|ty_descriptor| ty_descriptor.type_flags.contains(ReflectTypeFlags::STRUCT))
        .cloned()
        .collect::<Vec<_>>();
    let structs = {
        let mut ret = Vec::new();
        for stct in structs {
            // TODO: This logic should **probably** be recursive, but I doubt I will even actually need that
            if stct
                .decoration_flags
                .contains(ReflectDecorationFlags::BUFFER_BLOCK)
            {
                for rec in stct.members {
                    if rec.type_flags.contains(ReflectTypeFlags::STRUCT) {
                        ret.push(rec);
                    }
                }
            } else {
                ret.push(stct);
            }
        }
        ret
    };
    structs
        .iter()
        .map(|stct| {
            let name = stct.type_name.clone();
            let members = stct
                .members
                .iter()
                .map(|s| {
                    let name = s.struct_member_name.clone();
                    let ty = {
                        if s.type_flags
                            .contains(ReflectTypeFlags::FLOAT | ReflectTypeFlags::VECTOR)
                        {
                            // TODO: Atm only support `vec2, vec3, and vec4`
                            let num_components = s.traits.numeric.vector.component_count;
                            let row_count = s.traits.numeric.matrix.row_count;
                            let column_count = s.traits.numeric.matrix.column_count;
                            match (num_components, row_count, column_count) {
                                (2, 0, 0) => ShaderStructType::Vec2,
                                (3, 0, 0) => ShaderStructType::Vec3,
                                (4, 0, 0) => ShaderStructType::Vec4,
                                (3, 3, 3) => ShaderStructType::Mat3,
                                (4, 4, 4) => ShaderStructType::Mat4,
                                _ => panic!(
                                    "Not implemented: components: {}, row: {}, column: {}",
                                    num_components, row_count, column_count
                                ),
                            }
                        } else {
                            // TODO: Can support more types later
                            unimplemented!()
                        }
                    };
                    StructMember { name, ty }
                })
                .collect::<Vec<_>>();
            ShaderStruct { name, members }
        })
        .collect::<Vec<_>>()
}

#[derive(Debug, Copy, Clone)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

impl From<ShaderStage> for shaderc::ShaderKind {
    fn from(stage: ShaderStage) -> Self {
        match stage {
            ShaderStage::Vertex => shaderc::ShaderKind::Vertex,
            ShaderStage::Fragment => shaderc::ShaderKind::Fragment,
            ShaderStage::Compute => shaderc::ShaderKind::Compute,
        }
    }
}

pub struct ShaderCompiler<'a> {
    compiler: shaderc::Compiler,
    options: shaderc::CompileOptions<'a>,
}

impl<'a> ShaderCompiler<'a> {
    pub fn new(
        env_version: EnvVersion,
        opt_level: OptimizationLevel,
        include_dir: Option<PathBuf>,
    ) -> Result<Self> {
        let compiler =
            shaderc::Compiler::new().ok_or(ShaderCompilationError::CompilerCreationFailed)?;
        let mut options = shaderc::CompileOptions::new()
            .ok_or(ShaderCompilationError::CompilerOptionsCreationFailed)?;
        options.set_target_env(shaderc::TargetEnv::Vulkan, env_version as u32);
        options.set_optimization_level(opt_level);
        options.add_macro_definition("EP", Some("main"));
        if let Some(include_dir) = include_dir {
            options.set_include_callback(move |name, _, _, _| {
                let file_path = include_dir.join(name);
                let contents = std::fs::read_to_string(&file_path).unwrap_or_else(|_| {
                    panic!("Could not read shader include at: {:?}", file_path)
                });
                Result::Ok(shaderc::ResolvedInclude {
                    resolved_name: name.to_string(),
                    content: contents,
                })
            });
        }
        Ok(Self { compiler, options })
    }

    pub fn compile_shader_with_output_path(
        &self,
        shader_path: impl AsRef<Path>,
        output_path: impl AsRef<Path>,
        shader_stage: ShaderStage,
    ) -> Result<()> {
        let shader_path = shader_path.as_ref();
        let output_path = output_path.as_ref();
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).map_err(ShaderCompilationError::CouldNotCreateDir)?;
        }
        let file_name = shader_path
            .file_name()
            .ok_or_else(|| ShaderCompilationError::NoFileName(shader_path.to_owned()))?
            .to_str()
            .ok_or_else(|| ShaderCompilationError::NonUtf8Path(shader_path.to_owned()))?;
        let code = fs::read_to_string(shader_path)?;

        let compiled_shader = self
            .compiler
            .compile_into_spirv(
                &code,
                shader_stage.into(),
                file_name,
                "main",
                Some(&self.options),
            )
            .map_err(ShaderCompilationError::CompilationFailed)?;

        std::fs::write(output_path, compiled_shader.as_binary_u8())?;
        Ok(())
    }

    pub fn compile_shader(
        &self,
        shader_path: impl AsRef<Path>,
        shader_stage: ShaderStage,
    ) -> Result<()> {
        let shader_path = shader_path.as_ref();
        let output_path = if let Some(parents) = shader_path.parent() {
            if let Some(file_name) = shader_path.file_name() {
                if let Some(file_name) = file_name.to_str() {
                    Ok(parents.join("spv").join(format!("{}.spv", file_name)))
                } else {
                    Err(ShaderCompilationError::CouldNotGetShaderOutputPath)
                }
            } else {
                Err(ShaderCompilationError::CouldNotGetShaderOutputPath)
            }
        } else {
            Err(ShaderCompilationError::CouldNotGetShaderOutputPath)
        }?;
        self.compile_shader_with_output_path(shader_path, output_path, shader_stage)
    }
}

// TODO: This will be much better later when I actually turn this into a build-time syn thing
#[cfg(feature = "shader-structs")]
#[derive(Debug)]
pub enum ShaderStructType {
    Vec2,
    Vec3,
    Vec4,
    Mat3,
    Mat4,
}

#[cfg(feature = "shader-structs")]
#[derive(Debug)]
pub struct StructMember {
    pub name: String,
    pub ty: ShaderStructType,
}

#[cfg(feature = "shader-structs")]
#[derive(Debug)]
pub struct ShaderStruct {
    pub name: String,
    pub members: Vec<StructMember>,
}

#[cfg(feature = "shader-structs")]
#[derive(Debug)]
pub struct VertexAttribute {
    pub format: ReflectFormat,
    pub name: String,
    pub offset: u32,
    pub low_precision: bool,
}

#[cfg(feature = "shader-structs")]
#[derive(Debug)]
pub struct VertexAttributeDesc {
    pub stride: u32,
    pub atts: Vec<VertexAttribute>,
}

#[derive(Debug, Error)]
pub enum ShaderReflectionError {
    #[error("could not create shader module: {0}")]
    ShaderModuleCreationError(&'static str),
}

pub struct ShaderData {
    module: ShaderModule,
}

impl ShaderData {
    pub fn from_spv(spv_data: &[u8]) -> Result<Self, ShaderReflectionError> {
        let module = ShaderModule::load_u8_data(spv_data)
            .map_err(ShaderReflectionError::ShaderModuleCreationError)?;
        Ok(ShaderData { module })
    }

    pub fn module(&self) -> &ShaderModule {
        &self.module
    }

    pub fn stage(&self) -> ReflectShaderStageFlags {
        self.module.get_shader_stage()
    }

    pub fn source_file(&self) -> String {
        self.module.get_source_file()
    }

    // TODO: This is largely very bad
    #[cfg(feature = "shader-structs")]
    pub fn get_shader_structs(&self) -> Vec<ShaderStruct> {
        let blocks = self
            .module
            .enumerate_descriptor_bindings(None)
            .unwrap()
            .iter()
            .map(|binding| binding.block.clone())
            .collect::<Vec<_>>();

        get_structs_from_blocks(&blocks)
    }

    #[cfg(feature = "shader-structs")]
    pub fn get_push_constant_structs(&self) -> Vec<ShaderStruct> {
        let pc_blocks = self.module.enumerate_push_constant_blocks(None).unwrap();
        get_structs_from_blocks(&pc_blocks)
    }

    #[cfg(feature = "shader-structs")]
    pub fn get_vertex_attributes(&self) -> VertexAttributeDesc {
        let mut variables = self.module.enumerate_input_variables(None).unwrap();
        variables.sort_by(|a, b| a.location.cmp(&b.location));
        let mut offset = 0;
        let atts = variables
            .into_iter()
            .filter(|var| var.name != "gl_VertexIndex")
            .map(|var| {
                let low_precision = is_low_precision(&var.name);
                let num_components = if low_precision {
                    var.numeric.vector.component_count / 4
                } else {
                    var.numeric.vector.component_count
                };
                let component_size = var.numeric.scalar.width;
                let size = (num_components * component_size).max(component_size);
                let att = VertexAttribute {
                    format: var.format,
                    name: var.name,
                    offset,
                    low_precision,
                };
                offset += size;
                att
            })
            .collect::<Vec<_>>();
        VertexAttributeDesc {
            stride: offset,
            atts,
        }
    }
}

#[cfg(feature = "shader-structs")]
#[derive(Debug, Error)]
pub enum ShaderStructError {
    #[error("Could not create directory: {0}")]
    CouldNotCreateDir(std::io::Error),
    #[error(transparent)]
    IoError(#[from] std::io::Error),
}

#[cfg(feature = "shader-structs")]
pub fn shader_struct_to_rust(
    struct_name: &str,
    shader_struct: &ShaderStruct,
    archive: bool,
) -> syn::ItemStruct {
    let fields = shader_struct
        .members
        .iter()
        .map(|member| {
            let ty: syn::Type = match member.ty {
                ShaderStructType::Vec2 => parse_quote!([f32; 2]),
                ShaderStructType::Vec3 => parse_quote!([f32; 3]),
                ShaderStructType::Vec4 => parse_quote!([f32; 4]),
                ShaderStructType::Mat3 => parse_quote!([[f32; 3]; 3]),
                ShaderStructType::Mat4 => parse_quote!([[f32; 4]; 4]),
            };
            let ident = syn::Ident::new(&member.name, proc_macro2::Span::call_site());
            field_from_ident_and_type(ident, ty)
        })
        .collect::<Vec<_>>();

    struct_from_fields(struct_name, &fields, archive)
}

#[cfg(feature = "shader-structs")]
pub fn vertex_attributes_to_struct(
    struct_name: &str,
    attributes: &[VertexAttribute],
    archive: bool,
) -> syn::ItemStruct {
    let fields = attributes
        .iter()
        .map(|att| {
            let is_low_precision = is_low_precision(&att.name);
            let ty: syn::Type = match (att.format, is_low_precision) {
                (ReflectFormat::R32_UINT, _) => parse_quote!(u32),
                (ReflectFormat::R32_SINT, _) => parse_quote!(i32),
                (ReflectFormat::R32_SFLOAT, _) => parse_quote!(f32),
                (ReflectFormat::R32G32_UINT, _) => parse_quote!([u32; 2]),
                (ReflectFormat::R32G32_SINT, _) => parse_quote!([i32; 2]),
                (ReflectFormat::R32G32B32_UINT, _) => parse_quote!([u32; 3]),
                (ReflectFormat::R32G32B32_SINT, _) => parse_quote!([i32; 3]),
                (ReflectFormat::R32G32B32A32_UINT, _) => parse_quote!([u32; 4]),
                (ReflectFormat::R32G32B32A32_SINT, _) => parse_quote!([i32; 4]),
                (ReflectFormat::R32G32_SFLOAT, false) => parse_quote!([f32; 2]),
                (ReflectFormat::R32G32B32_SFLOAT, false) => parse_quote!([f32; 3]),
                (ReflectFormat::R32G32B32A32_SFLOAT, false) => parse_quote!([f32; 4]),
                (ReflectFormat::R32G32_SFLOAT, true) => parse_quote!([u8; 2]),
                (ReflectFormat::R32G32B32_SFLOAT, true) => parse_quote!([u8; 3]),
                (ReflectFormat::R32G32B32A32_SFLOAT, true) => parse_quote!([u8; 4]),
                _ => {
                    panic!("unsupported format: {:#?}", att)
                }
            };
            let ident = syn::Ident::new(&att.name, proc_macro2::Span::call_site());
            field_from_ident_and_type(ident, ty)
        })
        .collect::<Vec<_>>();

    struct_from_fields(struct_name, &fields, archive)
}

#[cfg(feature = "shader-structs")]
pub fn structs_to_file(
    path: impl AsRef<Path>,
    structs: &[syn::ItemStruct],
) -> Result<(), ShaderStructError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(ShaderStructError::CouldNotCreateDir)?;
    }
    let file: syn::File = parse_quote! {
        #(#structs)*
    };
    let formatted = prettyplease::unparse(&file);
    std::fs::write(path, &formatted)?;
    Ok(())
}

#[cfg(feature = "shader-structs")]
pub fn standardized_struct_name(prefix: &str, name: &str) -> String {
    use heck::ToUpperCamelCase;
    format!("{}_{}", prefix, name).to_upper_camel_case()
}
