## Using this template

- Edit pyproject.toml
  - set the description and version elements under the [project] table.
  - create a python package matching the project name under src. 
    - This is required to make UV work properly
  - Add the correct authors under [tool.hatch.metadata] in the authors block
  for example:
      ```toml
        [tool.hatch.metadata]
        authors = [
        { name = "Steven Fines", email = "stfines@corelogic.com" },
        ]
      ```
  - Update [tool.hatch.build.targets.wheel] with the root source package
    ```toml
      [tool.hatch.build.targets.wheel]
      packages = ["src/<your_project_root>", "src/<other project root>"]
    ```

