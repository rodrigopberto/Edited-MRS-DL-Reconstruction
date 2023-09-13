

input_path = "C:\Users\rodrigo\Documents\thesis\data\gannet_full_load";
output_path = "C:\Users\rodrigo\Documents\thesis\data\gannet_full_fit";

ge_site_list = ["g4","g6","g7","g8"];
ge_site_list = ["g5"];

ge_input_output_list = [];

for i=1:12
    for j=1:length(ge_site_list)
        sub_n_id = num2str(i,'%02d');
        sub_id = "S"+sub_n_id;
        input_filepath = input_path+"\"+ge_site_list(j)+"_"+lower(sub_id)+".mat";
        output_filepath = output_path+"\"+ge_site_list(j)+"_"+lower(sub_id)+".mat";
        ge_input_output_list = [ge_input_output_list, [input_filepath;output_filepath]];
    end
end

ph_site_list = ["p3","p4","p6","p7","p8","p9","p10"];

ph_input_output_list = [];

for i=1:12
    for j=1:length(ph_site_list)
        sub_n_id = num2str(i,'%02d');
        sub_id = "S"+sub_n_id;
        input_filepath = input_path+"\"+ph_site_list(j)+"_"+lower(sub_id)+".mat";
        output_filepath = output_path+"\"+ph_site_list(j)+"_"+lower(sub_id)+".mat";
        ph_input_output_list = [ph_input_output_list, [input_filepath;output_filepath]];
    end
end

si_site_list = ["s1","s3","s5","s6","s8"];

si_input_output_list = [];

for i=1:12
    for j=1:length(si_site_list)
        sub_n_id = num2str(i,'%02d');
        sub_id = "S"+sub_n_id;
        input_filepath = input_path+"\"+si_site_list(j)+"_"+lower(sub_id)+".mat";
        output_filepath = output_path+"\"+si_site_list(j)+"_"+lower(sub_id)+".mat";
        si_input_output_list = [si_input_output_list, [input_filepath;output_filepath]];
    end
end

%% GE Fit and Save

for i = 1:length(ge_input_output_list)
    input_filename = ge_input_output_list(1,i);
    output_filename = ge_input_output_list(2,i);
    mrs_struct = load(input_filename).mrs_struct;
    mrs_fit = GannetFitNoGraph(mrs_struct).out;
    save(output_filename,"mrs_fit");
end

%% PH LOADING AND SAVING

for i = 1:length(ph_input_output_list)
    input_filename = ph_input_output_list(1,i);
    output_filename = ph_input_output_list(2,i);
    mrs_struct = load(input_filename).mrs_struct;
    mrs_fit = GannetFitNoGraph(mrs_struct).out;
    save(output_filename,"mrs_fit");
end

%% SI Loading and Saving

for i = 1:length(si_input_output_list)
    input_filename = si_input_output_list(1,i);
    output_filename = si_input_output_list(2,i);
    mrs_struct = load(input_filename).mrs_struct;
    mrs_fit = GannetFitNoGraph(mrs_struct).out;
    save(output_filename,"mrs_fit");
end


