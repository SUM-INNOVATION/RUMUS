// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Procedural macros for the RUMUS deep learning framework.
//!
//! Provides `#[derive(Module)]` which auto-generates `parameters()`,
//! `train()`, `eval()`, `state_dict()`, and `load_state_dict()`
//! implementations by delegating to each named field's `Module` impl.
//! The user writes `forward()` as an inherent method — it is deliberately
//! **not** part of the `Module` trait.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

/// Derive the `Module` trait for a struct with named fields.
///
/// Generates:
/// - `parameters()` — concatenates `.parameters()` from every field.
/// - `train()` / `eval()` — delegates to every field.
/// - `state_dict(prefix)` — recursively collects tensors with dot-path keys.
/// - `load_state_dict(dict, prefix)` — recursively loads tensors by dot-path.
///
/// # Example
///
/// ```ignore
/// #[derive(Module)]
/// struct MLP {
///     linear1: Linear,
///     linear2: Linear,
/// }
///
/// impl MLP {
///     pub fn forward(&self, input: &Tensor) -> Tensor {
///         let x = self.linear1.forward(input);
///         let x = rumus::nn::relu(&x);
///         self.linear2.forward(&x)
///     }
/// }
/// ```
///
/// Calling `model.state_dict("")` produces keys like
/// `"linear1.weight"`, `"linear1.bias"`, `"linear2.weight"`, `"linear2.bias"`.
#[proc_macro_derive(Module)]
pub fn derive_module(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(named) => &named.named,
            _ => panic!("Module derive requires named fields"),
        },
        _ => panic!("Module derive only supports structs"),
    };

    let field_names: Vec<_> = fields
        .iter()
        .map(|f| f.ident.as_ref().expect("named field"))
        .collect();

    let expanded = quote! {
        impl #impl_generics rumus::nn::Module for #name #ty_generics #where_clause {
            fn parameters(&self) -> Vec<rumus::nn::Parameter> {
                let mut params = Vec::new();
                #(
                    params.extend(
                        rumus::nn::Module::parameters(&self.#field_names)
                    );
                )*
                params
            }

            fn train(&mut self) {
                #( rumus::nn::Module::train(&mut self.#field_names); )*
            }

            fn eval(&mut self) {
                #( rumus::nn::Module::eval(&mut self.#field_names); )*
            }

            fn state_dict(
                &self,
                prefix: &str,
            ) -> std::collections::HashMap<String, rumus::tensor::Tensor> {
                let mut dict = std::collections::HashMap::new();
                #(
                    dict.extend(
                        rumus::nn::Module::state_dict(
                            &self.#field_names,
                            &format!("{}{}.", prefix, stringify!(#field_names)),
                        )
                    );
                )*
                dict
            }

            fn load_state_dict(
                &mut self,
                dict: &std::collections::HashMap<String, rumus::tensor::Tensor>,
                prefix: &str,
            ) -> Result<(), rumus::autograd::AutogradError> {
                #(
                    rumus::nn::Module::load_state_dict(
                        &mut self.#field_names,
                        dict,
                        &format!("{}{}.", prefix, stringify!(#field_names)),
                    )?;
                )*
                Ok(())
            }
        }
    };

    expanded.into()
}
