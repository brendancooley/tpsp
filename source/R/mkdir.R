mkdir <- function(path) {
  if (!file.exists(path)) {
    dir.create(path)
  }
}