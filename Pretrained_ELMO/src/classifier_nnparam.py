from utils.imports import *
from utils.data_classifier import Dataset, DataLoader
from utils.preprocessing import preprocess
from utils.classifierr_model import LSTM, LSTMClassifier, ELMO_loaded

@torch.no_grad()
def get_accuracy(model, dataloader, config, embedding_model, save = False,type="test"):
    """
    Args:
        model: the model to train
        dataloader: the dataloader to get the data
        config: configuration of the model
    Returns:
        List of below metrics:
        loss: the loss of the model
        accuracy: the accuracy of the model
        micro_f1: the micro f1 score of the model
    """
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0
    model.eval()
    final_predictions = []
    final_targets = []
    final_accuracy = 0
    batch_num = 0
    # get batches from the dataloader
    for batch in dataloader:
        # get the input and target batch
        input_batch = batch[0]
        target_batch = batch[1]
        index = batch[2]
        # get the output of the model
        outputs = embedding_model(input_batch)
        layer1_output_forward = outputs["layer1_output_forward"]
        layer1_output_backward = outputs["layer1_output_backward"]
        layer2_output_forward = outputs["layer2_output_forward"]
        layer2_output_backward = outputs["layer2_output_backward"]
        embeddings = outputs["word_embeddings"]
        # concatenate the forward and backward outputs
        layer_output1 = torch.cat((layer1_output_forward, layer1_output_backward), dim=2)
        layer_output2 = torch.cat((layer2_output_forward, layer2_output_backward), dim=2)
        elmo_embedding = [embeddings, layer_output1, layer_output2]
        # get the predictions
        model_outputs = model(elmo_embedding)
        loss = loss_function(model_outputs["logits"], target_batch)
        normalised_logits = model_outputs["normalised_logits"]
        elmo_embeddings = model_outputs["elmo_embeddings"]
        if save:
            # save it inside the folder "/ssd_scratch/cvit/kolubex_anlp_elmo/model_name_classification/index"
            if not os.path.exists(f"/ssd_scratch/cvit/kolubex_anlp_elmo/{config['model_name_classification']}"):
                os.makedirs(f"/ssd_scratch/cvit/kolubex_anlp_elmo/{config['model_name_classification']}", exist_ok=True)
            torch.save(elmo_embeddings, f"/ssd_scratch/cvit/kolubex_anlp_elmo/{config['model_name_classification']}/{batch_num}.pkl")
        # get the predictions
        predictions = torch.argmax(normalised_logits, dim=1)
        # get the accuracy
        accuracy = torch.sum(predictions == target_batch)/len(target_batch)
        final_accuracy += accuracy
        final_targets.append(target_batch.cpu())
        final_predictions.append(predictions.cpu())
        # get all metrics like accuracy, precision, recall, f1 score, f1 micro, f1 macro
        # add the loss to the total loss
        total_loss += loss.item()
        batch_num += 1
    
    # print the classification report
    metrics = sk.metrics.classification_report(torch.cat(final_targets), torch.cat(final_predictions), output_dict=True)
    # append this metrics to the file with config a1, a2, a3 
    with open(f"./results/{config['model_name_classification']}.txt", "a") as f:
        f.write(str(metrics))
    # convert the metrics to a single level dictionary
    metrics_single_level = {}
    for key in metrics.keys():
        if metrics[key] == {}:
            for key1 in metrics[key].keys():
                metrics_single_level[type+key+key1] = metrics[key][key1]
        else:
            metrics_single_level[type+key] = metrics[key]
    # log the metrics to wandb
    wandb.log(metrics_single_level)            
    # get micro f1 score
    micro_f1 = metrics["weighted avg"]["f1-score"]
    return (total_loss/batch_num, final_accuracy/batch_num, micro_f1) 

def train(model, config, train_dataloader,val_dataloader,emedding_model):
    """
    Args:
        model: the model to train
        train_data: the training data
        epochs: number of epochs
        lr: learning rate
        batch_size: size of the batch
        config: configuration of the model
    Returns:
        model: the trained model
    """
    batch_size = config['batch_size']
    epochs = config['epochs']
    lr = config['lr']
    # initialize a LR scheduler on plateau
    if config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(),lr=lr)
    elif config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(),lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    loss_function = nn.CrossEntropyLoss()
    device = config['device']
    best_micro_f1 = 0
    best_model = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_num = 0
        # get batches from the dataloader
        for batch in train_dataloader:
            # get the input and target batch
            input_batch = batch[0]
            target_batch = batch[1]
            # zero the gradients
            model.zero_grad()
            # get the output of the model
            outputs = emedding_model(input_batch)
            layer1_output_forward = outputs["layer1_output_forward"]
            layer1_output_backward = outputs["layer1_output_backward"]
            layer2_output_forward = outputs["layer2_output_forward"]
            layer2_output_backward = outputs["layer2_output_backward"]
            embeddings = outputs["word_embeddings"]
            # concatenate the forward and backward outputs
            layer_output1 = torch.cat((layer1_output_forward, layer1_output_backward), dim=2)
            layer_output2 = torch.cat((layer2_output_forward, layer2_output_backward), dim=2)
            elmo_embedding = [embeddings, layer_output1, layer_output2]
            # get the predictions
            model_outputs = model(elmo_embedding)
            loss = loss_function(model_outputs["logits"], target_batch)
            # backpropagate the loss
            loss.backward()
            # update the parameters
            optimizer.step()
            # add the loss to the total loss
            total_loss += loss.item()
            # delete the outputs from the gpu memory
            del outputs
            batch_num += 1
        print("Epoch: {}, Loss: {}".format(epoch, total_loss/batch_num))
        if epoch == config['epochs']-1:
            saving_embeddings = True
        else:
            saving_embeddings = False
        val_loss, val_accuracy,val_micro_f1 = get_accuracy(model, val_dataloader,config, emedding_model,save = saving_embeddings,type="val")
        train_loss, train_accuracy,train_micro_f1 = get_accuracy(model, train_dataloader,config, emedding_model, save = saving_embeddings,type="train")
        print(f"Loss on the val set: {val_accuracy}")
        total_loss = total_loss/batch_num
        data_to_log = {
            "lr": optimizer.param_groups[0]['lr'],
            "epoch": epoch,
            "train_loss": train_loss,
            "train_micro_f1": train_micro_f1,
            "train_accuracy": train_accuracy.item(),
            "val_loss": val_loss,
            "val_micro_f1": val_micro_f1,
            "val_accuracy": val_accuracy.item(),         
        }
        print(data_to_log)
        wandb.log(data_to_log)
        if not os.path.exists("/ssd_scratch/cvit/kolubex_anlp_elmo/"):
            os.makedirs("/ssd_scratch/cvit/kolubex_anlp_elmo/")
        if val_micro_f1  > best_micro_f1:
            best_micro_f1 = val_micro_f1
            # save the model
            torch.save(model.state_dict(), f"/ssd_scratch/cvit/kolubex_anlp_elmo/{config['model_name_classification']}_best.pth")
            best_model = copy.deepcopy(model)
            print("Model saved!")
    return best_model.to(device)

def get_data_dict(data):
    toal_lines = len(data)
    data_dict = {}
    for i in range(len(data)):
        data[i] = data[i].strip()
        data[i] = data[i].lower().split(",")
        # merge from data[i][1] to len(data[i])
        data[i][1] = ",".join(data[i][1:])
        data[i][1] = preprocess(data[i][1])
        data_dict[i] = [data[i][1], data[i][0]]
    return data_dict

def edit_config(config, config1):
    config["hardcoded"] = config1["hardcoded"]
    config["a1"] = config1["weights"][0]
    config["a2"] = config1["weights"][1]
    config["a3"] = config1["weights"][2]
    config["output_dim"] = config["num_classes"]
    return config

def classification(config):
    train_file = open("./data/train.csv", "r")
    test_file = open("./data/test.csv", "r")
    # train_file = open("./data/testing.csv", "r")
    # test_file = open("./data/testing.csv", "r")
    print("Files opened!")
    train_data = train_file.readlines()
    test_data = test_file.readlines()
    # skip the first line
    train_data = train_data[1:]
    test_data = test_data[1:]
    # randomly select 5 data as test data
    random.seed(config['seed'])
    # keep first num_val_samples as val data
    val_data = train_data[:config['num_val_samples']]
    # val_data = train_data[:10]
    train_data = list(set(train_data) - set(val_data))
    test_data = test_data[1:]
    train_data_dict = get_data_dict(train_data)
    val_data_dict = get_data_dict(val_data)
    test_data_dict = get_data_dict(test_data)
    embedding_model = ELMO_loaded(config)
    embedding_model.to(config['device'])
    # create the dataset
    train_dataset = Dataset(train_data_dict,config)
    val_dataset = Dataset(val_data_dict,config)
    test_dataset = Dataset(test_data_dict,config)
    # create the dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    # create the model
    model = LSTMClassifier(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # train the model
    model = train(model,config, train_dataloader, val_dataloader, embedding_model)
    # test the model
    get_accuracy(model, test_dataloader, config, embedding_model, save = True,type="test")


# print("Starting the training!")

# with open("./elmo_test_config.json", "r") as f:
#     config = json.load(f)
# classification(config)


# def sweep_agent_manager():
#     wandb.init()
#     config1 = dict(wandb.config)
#     print(config1)
#     with open("./config.json", "r") as f:
#         config = json.load(f)
#     config = edit_config(config, config1)
#     config['model_name_classification'] = f"CLASSIFIER_{config['a1']}_{config['a2']}_{config['a3']}_{config['lr']}_{config['batch_size']}_{config['epochs']}_{config['optimizer']}_{config['seed']}_{config['num_val_samples']}_{config['num_layers']}_{config['bidirectional']}_{config['hardcoded']}"
#     run_name = config['model_name_classification']
#     wandb.run.name = run_name
#     classification(config)


if __name__ == "__main__":
    wandb.login()
    # wandb.agent(sweep_id="lakshmipathi-balaji/anlp_a2/0seklf1b", function=sweep_agent_manager, count=100)
    with open("./config.json", "r") as f:
        config = json.load(f)
    config1 = {
        "hardcoded": False,
        "weights": [0.3,0.3,0.3]
    }
    config = edit_config(config, config1)
    config['model_name_classification'] = f"CLASSIFIER_learanable_{config['lr']}_{config['batch_size']}_{config['epochs']}_{config['optimizer']}_{config['seed']}_{config['num_val_samples']}_{config['num_layers']}_{config['bidirectional']}_{config['hardcoded']}"
    run_name = config['model_name_classification']
    wandb.init(project="anlp_a2", entity="lakshmipathi-balaji", name=run_name, config=config, reinit=True)
    wandb.run.name = run_name
    classification(config)