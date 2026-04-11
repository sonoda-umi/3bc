
desc "Run experiments"
task :run do |t|
    sh <<~SH
      ./run_experiment_flow.sh
    SH
end

task :show do |t|
    sh <<~SH
        tree -L 1 figures
        tree -L 1 stats_output
    SH
end
