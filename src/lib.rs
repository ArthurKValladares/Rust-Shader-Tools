use anyhow::Result;
pub use shaderc::{EnvVersion, OptimizationLevel};
use std::{
    fs,
    path::{Path, PathBuf},
};
use thiserror::Error;

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
