import os.path
from pathlib import Path

import click
from PIL import Image, ImageFile, UnidentifiedImageError
from . import interrogate
import hydrus_api
from io import BytesIO
import json

Image.MAX_IMAGE_PIXELS = None

kaomojis = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]


@click.group()
def cli():
    pass


@click.command()
@click.argument("filename")
@click.option("--cpu", default=False, help="Use CPU instead of GPU")
@click.option(
    "--model", default="wd-v1-4-vit-tagger-v2", help="The tagging model to use"
)
@click.option("--threshold", default=0.35, help="The threshhold to drop tags below")
def evaluate(filename, cpu, model, threshold):
    if not os.path.isfile("./model/" + model + "/info.json"):
        raise ValueError("info.json not found in model folder!")

    with open("./model/" + model + "/info.json") as json_f:
        modelinfo = json.load(json_f)

    integerator = interrogate.WaifuDiffusionInterrogator(
        modelinfo["modelname"],  # the name of the model for display purposes
        modelinfo["modelfile"],  # the filename of the model file
        modelinfo["tagsfile"],  # the filename of the tags file
        model,  # the folder storing the previous two files as well as the info file
        modelinfo[
            "ratingsflag"
        ],  # flag indicating whether model identifies content rating
        modelinfo[
            "numberofratings"
        ],  # amount of tags to consider for content rating if so
        repo_id=modelinfo["source"],  # source of the model, credit where credit is due
    )
    integerator.load(cpu)
    image = Image.open(filename)
    ratings, tags = integerator.interrogate(image)
    rating = "none"
    if modelinfo["ratingsflag"]:
        ratings["none"] = (
            0.0  # assign none a value of zero so that rating comparison can still occur
        )
        for key in ratings.keys():
            if ratings[key] > ratings[rating]:
                rating = key
    clipped_tags = []

    for key in tags.keys():
        if tags[key] > threshold:
            clipped_tags.append(key)
    click.echo("rating: " + rating)
    click.echo("tags: " + ", ".join(clipped_tags))


@click.command()
@click.argument("path")
@click.option("--cpu", default=False, help="Use CPU instead of GPU")
@click.option(
    "--model", default="wd-v1-4-vit-tagger-v2", help="The tagging model to use"
)
@click.option("--threshold", default=0.35, help="The threshhold to drop tags below")
def evaluate_path(path, cpu, model, threshold):
    path = Path(path)
    click.echo(path)

    if not os.path.isfile("./model/" + model + "/info.json"):
        raise ValueError("info.json not found in model folder!")

    with open("./model/" + model + "/info.json") as json_f:
        modelinfo = json.load(json_f)

    integerator = interrogate.WaifuDiffusionInterrogator(
        modelinfo["modelname"],  # the name of the model for display purposes
        modelinfo["modelfile"],  # the filename of the model file
        modelinfo["tagsfile"],  # the filename of the tags file
        model,  # the folder storing the previous two files as well as the info file
        modelinfo[
            "ratingsflag"
        ],  # flag indicating whether model identifies content rating
        modelinfo[
            "numberofratings"
        ],  # amount of tags to consider for content rating if so
        repo_id=modelinfo["source"],  # source of the model, credit where credit is due
    )
    integerator.load(cpu)

    valid_ext = (".webp", ".png", ".jpg", ".jpeg")
    for file in path.iterdir():
        click.echo(f"{file.name.lower()=}")
        click.echo(f"{os.path.isfile(file)=}")

        if not file.is_file():
            continue

        if not file.name.lower().endswith(valid_ext):
            continue

        click.echo(f"processing {file}")

        image = Image.open(file)
        ratings, tags = integerator.interrogate(image)
        rating = "none"
        if modelinfo["ratingsflag"]:
            ratings["none"] = (
                0.0  # assign none a value of zero so that rating comparison can still occur
            )
            for key in ratings.keys():
                if ratings[key] > ratings[rating]:
                    rating = key
        clipped_tags = []

        for key in tags.keys():
            if tags[key] > threshold:
                clipped_tags.append(key)
        click.echo("rating: " + rating)
        click.echo("tags: " + ", ".join(clipped_tags))

        sidecar = file.with_suffix(".txt")
        with open(sidecar, "w") as fp:
            fp.write(", ".join(clipped_tags))


@click.command()
@click.argument("hash")
@click.option("--token", help="The API token for your Hydrus server")
@click.option("--cpu", default=False, help="Use CPU instead of GPU")
@click.option(
    "--model",
    default="SmilingWolf/wd-v1-4-vit-tagger-v2",
    help="The tagging model to use",
)
@click.option("--threshold", default=0.35, help="The threshhold to drop tags below")
@click.option(
    "--host", default="http://127.0.0.1:45869", help="The URL for your Hydrus server "
)
@click.option(
    "--tag-service", default="A.I. Tags", help="The Hydrus tag service to add tags to"
)
@click.option(
    "--ratings-only", default=False, help="Strip all tags except for content rating"
)
@click.option("--privacy", default=True, help="hides the tag output from the cli")
def evaluate_api(
    hash, token, cpu, model, threshold, host, tag_service, ratings_only, privacy
):
    if not os.path.isfile("./model/" + model + "/info.json"):
        raise ValueError("info.json not found in model folder!")

    with open("./model/" + model + "/info.json") as json_f:
        modelinfo = json.load(json_f)

    if ratings_only and not modelinfo["ratingsflag"]:
        raise ValueError("--ratings-only set, but model does not support ratings!")

    integerator = interrogate.WaifuDiffusionInterrogator(
        modelinfo["modelname"],  # the name of the model for display purposes
        modelinfo["modelfile"],  # the filename of the model file
        modelinfo["tagsfile"],  # the filename of the tags file
        model,  # the folder storing the previous two files as well as the info file
        modelinfo[
            "ratingsflag"
        ],  # flag indicating whether model identifies content rating
        modelinfo[
            "numberofratings"
        ],  # amount of tags to consider for content rating if so
        repo_id=modelinfo["source"],  # source of the model, credit where credit is due
    )
    integerator.load(cpu)
    client = hydrus_api.Client(token, host)
    image_bytes = BytesIO(client.get_file(hash).content)
    image = Image.open(image_bytes)
    ratings, tags = integerator.interrogate(image)
    rating = "none"
    if modelinfo["ratingsflag"]:
        ratings["none"] = (
            0.0  # assign none a value of zero so that rating comparison can still occur
        )
        for key in ratings.keys():
            if ratings[key] > ratings[rating]:
                rating = key
    clipped_tags = []

    if not ratings_only:
        for key in tags.keys():
            if tags[key] > threshold:
                clipped_tags.append(
                    key.replace("_", " ") if key not in kaomojis else key
                )

    if not privacy:
        click.echo("rating: " + rating)
        click.echo("tags: " + ", ".join(clipped_tags))

    if modelinfo["ratingsflag"]:
        clipped_tags.append("rating:" + rating)
    if ratings_only:
        clipped_tags.append(
            "ratings only " + modelinfo["modelname"] + " ai generated tags"
        )  # create tag specifying that content tags were excluded
    else:
        clipped_tags.append(
            "ai_generated: " + modelinfo["modelname"] + " ai generated tags"
        )  # create tag from given model name
    client.add_tags(hashes=[hash], service_names_to_tags={tag_service: clipped_tags})


@click.command()
@click.argument("hashfile")
@click.option("--token", help="The API token for your Hydrus server")
@click.option("--cpu", default=False, help="Use CPU instead of GPU")
@click.option(
    "--model", default="wd-v1-4-vit-tagger-v2", help="The tagging model to use"
)
@click.option(
    "--general", default=0.35, help="The threshhold to drop general tags below"
)
@click.option(
    "--character", default=0.8, help="The threshhold to drop character tags below"
)
@click.option(
    "--host", default="http://127.0.0.1:45869", help="The URL for your Hydrus server "
)
@click.option(
    "--tag-service", default="A.I. Tags", help="The Hydrus tag service to add tags to"
)
@click.option(
    "--ratings-only", default=False, help="Strip all tags except for content rating"
)
@click.option("--privacy", default=True, help="hides the tag output from the cli")
def evaluate_api_batch(
    hashfile,
    token,
    cpu,
    model,
    general,
    character,
    host,
    tag_service,
    ratings_only,
    privacy,
):
    if not os.path.isfile(hashfile):
        raise ValueError("hashfile not found!")
    if not os.path.isfile("./model/" + model + "/info.json"):
        raise ValueError("info.json not found in model folder!")

    with open("./model/" + model + "/info.json") as json_f:
        modelinfo = json.load(json_f)

    if ratings_only and not modelinfo["ratingsflag"]:
        raise ValueError("--ratings-only set, but model does not support ratings!")

    integerator = interrogate.WaifuDiffusionInterrogator(
        modelinfo["modelname"],  # the name of the model for display purposes
        modelinfo["modelfile"],  # the filename of the model file
        modelinfo["tagsfile"],  # the filename of the tags file
        model,  # the folder storing the previous two files as well as the info file
        modelinfo[
            "ratingsflag"
        ],  # flag indicating whether model identifies content rating
        modelinfo[
            "numberofratings"
        ],  # amount of tags to consider for content rating if so
        repo_id=modelinfo["source"],  # source of the model, credit where credit is due
    )
    integerator.load(cpu)
    client = hydrus_api.Client(token, host)
    with open(hashfile) as hashfile_f:
        hashes = hashfile_f.readlines()

    model_name = modelinfo["modelname"].lower()
    signature = f"ai_tagged:{model_name} g{general*100:.0f} c{character*100:.0f}"
    unable = []
    with click.progressbar(hashes) as bar:
        unable = []
        for hash in bar:
            hash = hash.strip()
            service_id = (
                "f1454ce45d8c13972a6b4d0d36771aaae11305cab6fd90d77d0d5f12b08d05b9"
            )
            tags: list[str] = client.get_file_metadata([hash])[0]["tags"][service_id][
                "storage_tags"
            ].get("0")

            if tags:
                for tag in tags:
                    if model_name in tag:
                        # click.echo(f" {hash} already tagged\n skipping...")
                        continue

            # click.echo(" processing: " + hash)
            image_bytes = BytesIO(client.get_file(hash).content)
            try:
                image = Image.open(image_bytes)
            except (UnidentifiedImageError, OSError):
                # click.echo(f"{hash} is not a supported file format\n skipping...")
                unable.append(hash)
                continue
            ratings, general_tags, character_tags = integerator.interrogate(image)
            rating = "none"
            if modelinfo["ratingsflag"]:
                ratings["none"] = (
                    0.0  # assign none a value of zero so that rating comparison can still occur
                )
                for key in ratings.keys():
                    if ratings[key] > ratings[rating]:
                        rating = key
            clipped_tags = []

            if not ratings_only:
                for key in general_tags.keys():
                    if general_tags[key] > general:
                        clipped_tags.append(
                            key.replace("_", " ") if key not in kaomojis else key
                        )
                for key in character_tags.keys():
                    if character_tags[key] > character:
                        clipped_tags.append(
                            key.replace("_", " ") if key not in kaomojis else key
                        )

            if not privacy:
                click.echo("rating: " + rating)
                click.echo("tags: " + ", ".join(clipped_tags))
                click.echo()

            if modelinfo["ratingsflag"]:
                clipped_tags.append("rating:" + rating)
            if ratings_only:
                clipped_tags.append(
                    "ratings only " + signature
                )  # create tag specifying that content tags were excluded
            else:
                clipped_tags.append(signature)

            client.add_tags(
                hashes=[hash], service_names_to_tags={tag_service: clipped_tags}
            )

    unable_s = "\n".join(unable)
    click.echo(f"\nUnable to process hashes:\n{unable_s}")


if __name__ == "__main__":
    Image.init()
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    cli.add_command(evaluate)
    cli.add_command(evaluate_api)
    cli.add_command(evaluate_api_batch)
    cli.add_command(evaluate_path)
    cli()
