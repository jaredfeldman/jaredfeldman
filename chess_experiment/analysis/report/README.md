# Master .Rmd
- The .Rmd that will knit the entire report is called `chess_report_master.Rmd`.
- All libraries used throughout the report must be included in the `r load packages` chunk.
- `chess_report_master.Rmd` has each "Section" .Rmd ordered after the table of contents.
- If adding a new "Section" .Rmd, insert where it fits in the paper using the same format as existing ones.
 
# Section .Rmds
- Each section has it's own .Rmd file
- When making changes to "Section" .Rmds:
  - Add any new libraries used to the top of the `chess_report_master.Rmd` (this will ensure it knits properly)
  - Knit `chess_report_master.Rmd` often during edits to prevent breaking changes.
