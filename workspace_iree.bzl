"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
load("@iree_core//build_tools/bazel:workspace.bzl", "configure_iree_submodule_deps", "configure_iree_cuda_deps")

def workspace():
    # skylib
    http_archive(
        name = "bazel_skylib",
        sha256 = "66ffd9315665bfaafc96b52278f57c7e2dd09f5ede279ea6d39b2be471e7e3aa",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.4.2/bazel-skylib-1.4.2.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.4.2/bazel-skylib-1.4.2.tar.gz",
        ],
    )

    bazel_skylib_workspace()


    # llvm-project
    # FIXME: This is supposed to be handled by the iree workspace?
    # How didn't the pjrt plugin show an error without this?
    # new_local_repository(
    #   name = "llvm-raw",
    #   build_file_content = "# empty",
    #   path = "../iree/third_party/llvm-project",
    # )
    # llvm_configure(name = "llvm-project")


    configure_iree_submodule_deps(
        iree_repo_alias = "@iree_core",
        iree_path = "../iree",
    )

    configure_iree_cuda_deps()


# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
iree_workspace = workspace
