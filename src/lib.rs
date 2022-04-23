use anyhow::Result;
use std::{fs, path::Path};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ShaderCompilationError {
    #[error("could not create shader compiler")]
    CompilerCreationFailed,
    #[error("could not create shader compiler options")]
    CompilerOptionsCreationFailed,
    #[error("the shader path is not a valid UTF-8 string")]
    NonUtf8Path(std::path::PathBuf),
    #[error("path does not contain a valid file name")]
    NoFileName(std::path::PathBuf),
}

pub struct ShaderCompiler<'a> {
    compiler: shaderc::Compiler,
    options: shaderc::CompileOptions<'a>,
}

impl<'a> ShaderCompiler<'a> {
    pub fn new() -> Result<Self> {
        let compiler =
            shaderc::Compiler::new().ok_or(ShaderCompilationError::CompilerCreationFailed)?;
        let mut options = shaderc::CompileOptions::new()
            .ok_or(ShaderCompilationError::CompilerOptionsCreationFailed)?;
        options.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_2 as u32,
        );
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);
        options.add_macro_definition("EP", Some("main"));

        Ok(Self { compiler, options })
    }

    pub fn compile_shader(
        &self,
        shader_path: impl AsRef<Path>,
        output_path: impl AsRef<Path>,
    ) -> Result<()> {
        let shader_path = shader_path.as_ref();
        let output_path = output_path.as_ref();

        let file_name = shader_path
            .file_name()
            .ok_or_else(|| ShaderCompilationError::NoFileName(shader_path.to_owned()))?
            .to_str()
            .ok_or_else(|| ShaderCompilationError::NonUtf8Path(shader_path.to_owned()))?;
        let code = fs::read_to_string(shader_path)?;

        // TODO: better wayt to get shader kind, hook up additional_options
        let compiled_shader = self.compiler.compile_into_spirv(
            &code,
            shaderc::ShaderKind::InferFromSource,
            file_name,
            "main",
            Some(&self.options),
        )?;

        std::fs::write(&output_path, compiled_shader.as_binary_u8())?;

        Ok(())
    }
}
