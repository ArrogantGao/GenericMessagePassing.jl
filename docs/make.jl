using GenericMessagePassing
using Documenter

DocMeta.setdocmeta!(GenericMessagePassing, :DocTestSetup, :(using GenericMessagePassing); recursive=true)

makedocs(;
    modules=[GenericMessagePassing],
    authors="Xuanzhao Gao <gaoxuanzhao@gmail.com> and contributors",
    sitename="GenericMessagePassing.jl",
    format=Documenter.HTML(;
        canonical="https://ArrogantGao.github.io/GenericMessagePassing.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ArrogantGao/GenericMessagePassing.jl",
    devbranch="main",
)
