use anyhow::Result;
use spirv_reflect::{
    types::{ReflectDecorationFlags, ReflectTypeFlags},
    ShaderModule,
};
use std::{
    fs,
    path::{Path, PathBuf},
};
use thiserror::Error;
pub use {
    shaderc::{EnvVersion, OptimizationLevel},
    spirv_reflect::types::{descriptor::ReflectDescriptorType, variable::ReflectShaderStageFlags},
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

#[derive(Debug, Copy, Clone)]
pub enum ShaderStage {
    Vertex,
    Fragment,
}

impl From<ShaderStage> for shaderc::ShaderKind {
    fn from(stage: ShaderStage) -> Self {
        match stage {
            ShaderStage::Vertex => shaderc::ShaderKind::Vertex,
            ShaderStage::Fragment => shaderc::ShaderKind::Fragment,
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
#[derive(Debug)]
pub enum ShaderStructType {
    Vec2,
    Vec3,
    Vec4,
}

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
        let ty_descriptors = self
            .module
            .enumerate_descriptor_bindings(None)
            .unwrap()
            .iter()
            .map(|binding| &binding.block)
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
                                let num = s.traits.numeric.vector.component_count;
                                match num {
                                    2 => ShaderStructType::Vec2,
                                    3 => ShaderStructType::Vec3,
                                    4 => ShaderStructType::Vec4,
                                    _ => unimplemented!(),
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
}

#[cfg(feature = "shader-structs")]
pub fn shader_struct_to_rust(struct_name: &str, shader_struct: &ShaderStruct) -> syn::ItemStruct {
    use syn::parse_quote;

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

    let fields = shader_struct
        .members
        .iter()
        .map(|member| {
            let ty: syn::Type = match member.ty {
                ShaderStructType::Vec2 => parse_quote!([f32; 2]),
                ShaderStructType::Vec3 => parse_quote!([f32; 3]),
                ShaderStructType::Vec4 => parse_quote!([f32; 4]),
            };
            let ident = syn::Ident::new(&member.name, proc_macro2::Span::call_site());
            field_from_ident_and_type(ident, ty)
        })
        .collect::<Vec<_>>();

    let struct_ident = syn::Ident::new(struct_name, proc_macro2::Span::call_site());
    parse_quote! {
        #[repr(C)]
        #[derive(Debug)]
        pub struct Test {
            #(#fields,)*
        }
    }
}

#[cfg(feature = "shader-structs")]
pub fn standardized_struct_name(shader_source_file: &str, type_name: &str) -> String {
    use heck::ToUpperCamelCase;
    format!("{}{}", shader_source_file, type_name).to_upper_camel_case()
}
