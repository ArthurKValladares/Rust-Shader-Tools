use thiserror::Error;

#[derive(Debug, Error)]
pub enum ShaderCompilationError {
    #[error("the shader path is not a valid UTF-8 string")]
    NonUtf8Path(std::path::PathBuf),
}

pub struct ShaderCompiler {
    compiler: shaderc::Compiler,
    options: shaderc::CompileOptions,
}

impl ShaderCompiler {
    pub fn new() -> Result<Self> {
        let compiler = shaderc::Compiler::new()?;
        let options = shaderc::CompileOptions::new()?;

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
            .ok_or_else(|| ShaderCompilationError::NonUtf8Path(shader_path.to_owned()))?;
        let code = fs::read_to_string(shader_path)?;

        // TODO: better wayt to get shader kind, hook up additional_options
        let compiled_shader = self.compiler.compile_into_spirv(
            &code,
            shaderc::ShaderKind::InferFromSource,
            &file_name,
            "main",
            None,
        )?;

        std::fs::write(&output_path, compiled_shader.as_binary_u8())?;

        Ok(())
    }
}
